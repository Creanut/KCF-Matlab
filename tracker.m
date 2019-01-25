function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%   KCF算法中的“核化”指的是使用的非线性的核，DCF中选用的是线型核
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014


	%if the target is large, lower the resolution, we don't need that much
	%detail
    %如果待处理的帧图像的尺寸过大，则直接将其缩小到原来的一般，同时框选出的目标
    %位置也同比例缩小。这里设置的“大图片帧”的判断阈值是超过10000个像素点。
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end


	%window size, taking padding into account
    %确定padding框的大小。
	window_sz = floor(target_sz * (1 + padding));
	
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
    %贴标签使用的是高斯函数的形状，即越靠近跟踪目标的区域，其标签值越大，即表明
    %其在后续的回归器训练过程中，这部分的权重越高。
    %对样本标签y进行离散傅里叶变换，此处使用fft2目的是得到标签的回归值。

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
    %对样本加窗处理是防止在频域出现频谱泄露的情况发生，即滤除边界效应，这里的
    %余弦窗是根据yf的大小得出的，两个汉宁窗相乘得到的是一个立体的余弦窗。
	
	
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.
    %变量以f结尾表示在傅里叶域内

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    %这里的numel(img_files)表示的是视频的帧数，初始化为帧数*2的矩阵，目的是用来
    %存放每一帧计算出来的最大响应的位置。

    %逐帧读取图像
	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
        %彩色图像转换为灰度值图像
		if size(im,3) > 1,
			im = rgb2gray(im);
        end
        %目标过大，则缩小至原来的一半
		if resize_image,
			im = imresize(im, 0.5);
		end

		tic()
        %开始计时
        %------------------------下面为推理阶段----------------------------
        %操作目标为测试帧，即根据上一帧来计算这一帧图像的
        %相关响应值
		if frame > 1,
            %这需要通过上一帧的结果来获取子窗口，并将测试帧转换到傅里叶域
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			patch = get_subwindow(im, pos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			
			%calculate response of the classifier at all shifts
            %计算每个循环移位生成的样本在回归器中的响应值
            %在计算之前需要根据选择的核类型对当前帧zf和上一帧得到的model_xf进行
            %核相关计算得到kzf，然后再与回归器得到的权重系数进行点乘运算，得到
            %上一帧训练出的回归器在这一帧中运算得到的响应值。
			switch kernel.type
			case 'gaussian',
				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
			case 'polynomial',
				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
			case 'linear',
				kzf = linear_correlation(zf, model_xf);
			end
			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            %只保留结果的实部

			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
            %经过高斯化并且将标签值进行移位后，标签矩阵的左上角为原样本的标签值，
            %正常情况下，该标签值应该为1
            %此处寻找在新的样本下，响应值最大的位置，该位置即为本帧图像中目标所在的位置
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
            %这里是找到响应的最大位置
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
            %更新目标位置
        end
        %-------------截止到这里，后续帧图像中不断迭代的步骤结束。-----------

        %----------------------以下为回归器的训练过程-----------------------
		%obtain a subwindow for training at newly estimated target position
        %获取目标的位置和窗口大小
		patch = get_subwindow(im, pos, window_sz);
        %x为训练过程中输入的训练样本，xf是将样本数据转化到傅里叶域的结果
		xf = fft2(get_features(patch, features, cell_size, cos_window));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
        %这里为论文公式中计算alpha的过程，其中会根据所使用的核函数的类型首先
        %计算Kxx的值。
		switch kernel.type
		case 'gaussian',
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training

        %如果是第一帧图像，则它的结果就是直接训练得到的权值和模板
		if frame == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model
            %在后续帧中，权值和模板是当前帧的训练结果和上一次的训练结果的互补融合
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
		end

		%save position and timing
        %保存每一帧中所检测到的目标的位置，处理算法的运行时间
		positions(frame,:) = pos;
		time = time + toc();

		%visualization
		if show_visualization,
            %在交互界面显示出所找到的目标
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
    end

    %将之前缩小的图像中得到的目标位置逆变换回去
	if resize_image,
		positions = positions * 2;
	end
end

