function [] = video2frames(path_inputvideo, path_outputfolder)
%% Author: Lili Cai, Last update: 2/21/2020
% This function takes an input video, creates a folder with that video
% name, and extracts all the frames for the video into that folder
%
% Inputs:   
% path_inputvideo     full path to input video
% path_outputfolder   path to create output folder
%
% Example: 
% if your input path is: '/main/video1.mov'
% >> video2frames(fullfile('main', 'video1.mov'), fullfile('main'))
% --> use fullfile because the path separator will be different on Mac vs.
% Windows vs. Unix
% Will create a folder /main/video1, and in that folder will contain: 
% frame_00001.jpg
% frame_00002.jpg ... frame_XXXXX.jpg
% the frames of video1


%%
    % Upload video ====================================================
    % from the input string, extract video filename and file path
    videoname = strsplit(path_inputvideo, filesep);
    if length(videoname) >= 2
        ind = strfind(path_inputvideo, videoname{end});
        filepath = path_inputvideo(1:ind-1);
        full_filename = videoname{end};
    else
        filepath = ' ';
        full_filename = videoname{1};
    end
    filename = strsplit(full_filename,'.');
    filename = filename{1};
        
    try
        %filepath               % check point: display filepath
        %full_filename          % check point: display full_filename
        cd(filepath)            % cd to folder with video
        disp(['Uploading Video: ' full_filename '..... '])
        vid = VideoReader(full_filename);   % load video
    catch 
        disp(['Skipped, error uploading video: ' path_inputvideo])
    end

    % Make folder for frames + create frames  =========================
    cd(path_outputfolder)
    mkdir([filename '_frames'])                             % make a directory with that folder name
    cd([filename '_frames'])    % cd to that folder where frames will be stored
    
    ii = 1;                      % set frame number
    disp(['Video Processing to frames: ' filename])

	while hasFrame(vid)
	    img = readFrame(vid);
        frame_name = [filename '_' num2str(ii) '.jpg'];

        imwrite(img, frame_name); %Write out to a JPEG file
        ii = ii+1;
        if mod(ii,2000)==0
            disp(['Completed frame' num2str(ii)])
        end
	end
 
    disp(['Video to Frames Finished: ' filename ', ----------------------'])    
    cd ../../                      % cd to parent folder
    
    
end

