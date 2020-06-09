%% This is the demo script
% Author: Lili X. Cai, last update: 2/25/2020

% (1) Extract frames from video m004_00.mov  
%     (mouseID: m004, experiment day details: 00)
% (2) Create CNN images from each consecutive frame
% (3) Manually label random frames from extracted frames 1 or 0 
% (4) Take manually labeled frames and copy CNN images from manually
% labeled frames into either 'freeze' or 'nofreeze' folder

% Note: (2) or (3) can happen in either order, or in parallel
%% 
cur_dir = pwd;
%% (1) Extract frames from video
video2frames('m004_00.mov', '.')   % output path is '.' because we want output folder to be in the current folder
                                   % recommend using Matlab2016b,
                                   % VideoReader() might have bugs with
                                   % other Matlab versions
                                   
cd(cur_dir)                        % return to current directory
%% (2) Create CNN images from frames
frames2cnn('m004_00_frames', '.')  % default threshold is 15, so only put 2 arguments in
cd(cur_dir) 

%% (3) Label random consecutive frames 
scoring('Lili', 'm004_00_frames', 'scoring', 50, 3, 3)
cd(cur_dir) 
%% (4) Take labeled frames, put the CNN image version of those frames in either 'freeze' or 'nofreeze folder
scoring2cnn('scoring', '.', '.', 'cnn_trainingImages')
cd(cur_dir) 