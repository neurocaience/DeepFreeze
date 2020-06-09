function [] = frames2cnn(path_inputfolder, path_outputfolder, thresh)
%% 
% Author: Lili Cai, last update 2/24/2020
%
% This function transforms raw image frames from video into a CNN input
% frame. The input frame is the absolute pixel intensity difference between
% two consecutive frames. 
%
% Inputs: 
% path_inputfolder    ie. '\main\video1_frames'
% path_outputfolder   ie. '\main\'
% thresh              default = 15. This is 99% max pixel intensity

% Outputs: 
% Will create a folder 'video1_frames4cnn' in '\main' which holds all the
% CNN image files

% CUSTOMIZATION NOTES: 
% (1) Image cropping   
% This doesn't have any image cropping, but for the purposes of CNN, you
% might want to crop your image if there are irrelevant borders of the
% video. In this case, Ctrl+F for "%%%% CHANGE ME" and uncomment + put in
% the borders that you want to crop.
% (2) More than 99,000 frames in the video
% If you have more than 99,000 frames in the video, Ctrl+F for "%%%% CHANGE
% ME" and change the number of zeros that are padded to the beginning of
% the CNN image file

% FORMATTING NOTES: 
% (1) The inputfolder 
% Input folder should be formatted 'X_frames' where X is name of video
% (2) .jpg files in the Inputfolder 
% Jpg files in inputfolder should have this format: 
% mouseID_experimentID_#.jpg  ie. m45_00_1.jpg ... m45_00_45983.jpg
% Where # is frame # from video2frames
% (3) Output folder
% The outputfolder should be the main folder where you want to folder + CNN
% frames are to be make:
% ie. it will create the folder '\main\video1_frames4cnn'
%


if nargin <= 2
    thresh = 15;
end

%% Extract filename from path_inputfolder
foldername = strsplit(path_inputfolder, filesep);
%foldername
if length(foldername) >= 2
    ind = strfind(path_inputfolder, foldername{end});
    filepath = path_inputfolder(1:ind-1);
    foldername = foldername{end};
    filepath
    foldername
else
    filepath = ' ';
    foldername = foldername{1};
    filepath
    foldername
end
foldername_split = strsplit(foldername, '_');
ind = strfind(foldername, foldername_split{end});
file_pre = foldername(1:(ind-1));
%file_pre

out_folder = [foldername '_cnn'];
outpath = fullfile(path_outputfolder,out_folder);

%% Set whether you want the zscore or threshold. 
apply_zscore = 1; 
apply_thresh = 1;

%% Transform into CNN files

try
    mkdir(outpath)

    all_frames = dir(fullfile(path_inputfolder,'*.jpg')); % only get .jpg frames
    num_frames = size(all_frames,1);
    num_frames

    disp([foldername ': Start ---------'])
    for frame = 1:(num_frames-1)
        % check point: uncomment to check filenames
        %[fullfile(path_inputfolder, file_pre) num2str(frame) '.jpg']
        %[fullfile(path_inputfolder, file_pre) num2str(frame+1) '.jpg']
        
        % load images
        image1 = imread([fullfile(path_inputfolder, file_pre) num2str(frame) '.jpg']);
        image2 = imread([fullfile(path_inputfolder, file_pre) num2str(frame+1) '.jpg']);

        % Grayscale the frames + crop
        test1 = rgb2gray(image1);
        test2 = rgb2gray(image2);
        
        % UNCOMMENT TO CROP IMAGE
        %test1 = test1(1:230,50:279);    %%%% CHANGE ME crop into 230*230
        %test2 = test2(1:230,50:279);

        % Turn images into a double
        test1 = double(test1);           
        test2 = double(test2);

        % Take absolute value of pixel intensity difference between
        % consecutive frames
        temp = abs(test1 - test2);
        
        % Apply Z-score and thresholding
        if apply_zscore == 1
            temp = (temp - mean(temp(:)))/std(temp(:));
        end
        if apply_thresh == 1
            temp(temp >= thresh) = thresh;

            var_max = max(max(temp));
            var_mult = 255/var_max;
            temp = temp*var_mult;
        end  
        
        % Turn into uint8 format
        temp2 = uint8(temp);
        
        % Write CNN image to folder
        imwrite(temp2,fullfile(outpath,[foldername '_' num2str(frame,'%05d') '.png']), 'BitDepth', 16);   %%%% CHANGE ME if you have more frames than 99,000
        
        % Display progress every 1000 frames
        if mod(frame,1000) == 0
            disp(num2str(frame))
        end
    end
    disp([foldername ': COMPLETED'])

catch
  disp([foldername ': Did not run ============================== '])
end



end