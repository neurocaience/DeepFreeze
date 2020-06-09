function [] = scoring(name, path_foldertoscore, path_foldertosave, num_frames_to_score, first_frame, rand_seed)
%%  
% Manual labeling script for quick scoring
% 
% Author: Lili Cai & Cameron Hayes, last update: 2/24/2020
% This script has a GUI
% It lets a human observer score N random consecutive frames in a folder
% It saves the scoring in a .mat file with the info: 
%      (1) file name path of first image (ie.
%          Y:\Lili\random\m004_00_frames_cnn\m004_00_1932.jpg)
%      (2) 1 or 0 (for 'freeze' or 'no freeze')
%
% To score:  
%       1. Run this script. Either click 'Run' under Editor tab. OR Ctrl+Enter
%       2. Click on the 'figure'
%       3. Press a keyboard input: 
%          right arrow                 = next frame
%          left arrow                  = previous frame
%          up arrow OR '1' on numpad   = 1  'no freeze'
%          down arrow OR '0' on numpad = 0 'freeze'
% 

% Common troubleshooting issues: 
% (A) Your figure action is set to 'zoom in', 'zoom out', 'figure cursor' or 
% another action. Make sure any figure actions are deselected
% (B) Your input folder is the CNN folder instead of the raw frames folder
% 

if nargin <= 4
    first_frame = 2;
    rand_seed = 2;
elseif nargin <= 5
    rand_seed = 2;
end

%% Input variables ========================================================
%name = 'yourname';   % Your name
%path_foldertoscore = 'Y:\Lili\random\m004_00_frames';   % Folder with frames you want to score: 
%path_foldertosave = 'Y:\Lili\random\scoring';           % Folder where you want to save scored files
%rand_seed = 2;        % Random seed (so that these numbers can be reproducible)
%num_frames_to_score = 25; 
%first_frame = 100;      % Start random selection on this frame. Default = 3 

% =========================================================================
% Make directory for save path
mkdir(path_foldertosave)

% Extracting data
disp(['File: ' path_foldertoscore])

% Display + calculate number of frames in the folder
all_frames = dir(fullfile(path_foldertoscore,'*.jpg')); % only get .jpg frames
num_frames = size(all_frames,1);
num_frames

% Extract filename and file_prefix from path_inputfolder
foldername = strsplit(path_foldertoscore, filesep);
if length(foldername) >= 2
    ind = strfind(path_foldertoscore, foldername{end});
    filepath = path_foldertoscore(1:ind-1);
    foldername = foldername{end};
else
    filepath = ' ';
    foldername = foldername{1};
end
foldername_split = strsplit(foldername, '_');
%ind = strfind(foldername, foldername_split{end}); % deleted 3/4/2020
% file_pre = foldername(1:(ind-1));

ind = strfind(foldername, foldername_split{2});  % added 3/4/2020
file_pre = [foldername(ind:end) '_'];

foldername
foldername_split
file_pre

% Randomly select frames to score between first_frame and num_frame: 
rng(rand_seed);     % set the rand seed for reproducibility
frames_to_score = randperm(num_frames - first_frame, num_frames_to_score) + first_frame; 

% Create data matrices to store frame # that is scored and score 
data = {};   % data.frame{1} = 'm755_01_cs...'; data.frame{2} = 'm755_01_cs...'
             % data.val{1} = 1; data.val{0} = 0; cell2mat(data.val) = [1 0]
             % this stores the name of the frame being analyzed, in
             % addition to the score of the frame
scored = nan(num_frames_to_score,1);

% Do the actual scoring
global KEY_IS_PRESSED out   % set a global variable
KEY_IS_PRESSED = 0;

figure(1)
set(gcf, 'KeyPressFcn', @myKeyPressFcn)
        %j=1+blip; % + 14 here, and also in f below:
        f = 1;       % f is the new 'blip' 
        while (f <= length(frames_to_score))
            
            frame=frames_to_score(f);
            
            disp(['Index#: ' num2str(f) ', Frame#: ' num2str(frame)])

            % Open the file: 
            fn = fullfile(path_foldertoscore,[file_pre num2str(frame) '.jpg']);
            fn2 = fullfile(path_foldertoscore,[file_pre num2str(frame+1) '.jpg']);
            image1 = imread(fn);
            image2 = imread(fn2);   

            % Grayscale the frames + crop
            test1 = rgb2gray(image1);
            test2 = rgb2gray(image2);
            
            % %%%% CHANGE ME: if you want to crop the image
            %test2 = test2(1:230,60:279);
            %test1 = test1(1:230,60:279);    % crop into 230*220
            
            % Find the difference between frames
            diff = abs(test1-test2);
            
	    clf, 
            % Loop current and next file    
            while ~KEY_IS_PRESSED 
                imagesc(image1),colormap(gray);  text(10,10,['frame#' num2str(frame) ' ind#' num2str(f) ' score:' num2str(scored(f))],'color','r', 'fontsize',20); ...              
                    pause(.1)
                imagesc(image2),colormap(gray); text(10,10,['frame#' num2str(frame+1) ' ind#' num2str(f) ' score:' num2str(scored(f))],'color','r', 'fontsize',20); ...
                    pause(.1)
            end
            
            % If right arrow, go to next index, 
            % If left arrow, go to previous index: 
            if strcmpi(out, 'rightarrow')
                f = f+1;
            elseif strcmpi(out, 'leftarrow')
                f = f-1;
            else
            
                %Save input: Did it freeze (0) or was there movement (1)?
                if length(out) >= 6 && ~contains(out,'arrow')
                    sub = extractAfter(out,6);
                    if strcmpi(sub, '0') || strcmpi(sub, '1')
                        scored(f) = str2double(sub);
                        disp(['Score: ' num2str(scored(f))])
                        f = f+1;
                    end
                elseif strcmpi(out, '0') || strcmpi(out, '1')
                        scored(f) = str2double(out); 
                        disp(['Score: ' num2str(scored(f))])  
                        f = f+1;
                elseif strcmpi(out, 'uparrow')
                        scored(f) = 1;
                        disp(['Score: ' num2str(scored(f))])  
                        f = f+1;
                elseif strcmpi(out, 'downarrow')
                        scored(f) = 0;
                        disp(['Score: ' num2str(scored(f))])  
                        f = f+1;
                end
            end
            
            KEY_IS_PRESSED = 0;
            
        end
disp('Done Scoring')

% Save files
save(fullfile(path_foldertosave, [file_pre 'seed' num2str(rand_seed) '_' name '_scores']), 'scored')
save(fullfile(path_foldertosave, [file_pre 'seed' num2str(rand_seed) '_' name '_scoredframes']), 'frames_to_score')
disp('Saved.')
      
end

%%
function myKeyPressFcn(hObject, event)
 global KEY_IS_PRESSED out
 KEY_IS_PRESSED  = 1;
 out = event.Key;
 %disp('key is pressed') 
 %disp(event.Key)
end

