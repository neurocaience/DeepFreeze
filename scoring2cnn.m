function [] = scoring2cnn(path_folderwithscores, path_outputfolder, path_cnnimages, cnn_name)
%% 
% This script takes scored files and turns them into folders with 'freeze'
% and 'nofreeze' for input to CNN training
%
% Author: Lili X. Cai, last update 2/24/2020

% Input: ==========================================================
% Change your inputs, as relevant: 
%path_folderwithscores = 'Y:\Lili\random\scoring';           % Folder where you want to save scored files
%path_outputfolder = 'Y:\Lili\random';
%path_cnnimages = 'Y:\Lili\random';
%cnn_name = 'CNN_context1';        % Name of CNN
% =================================================================

% Make new folders
mkdir(fullfile(path_outputfolder, cnn_name))
mkdir(fullfile(path_outputfolder, cnn_name,'freeze'))
mkdir(fullfile(path_outputfolder, cnn_name, 'nofreeze'))

% Create listing of all scores and scoredframes .mat files
scores_list = dir(fullfile(path_folderwithscores, '*_scores.mat'));
scoredframes_list = dir(fullfile(path_folderwithscores, '*_scoredframes.mat')); 

% For each file in scoredframes_list, for each scored frame in that file, 
% assign to either 'freeze' or 'nofreeze' folder: 
for sf = 1:size(scoredframes_list,1)
    % If the file contains a .mat ending, process the file: 
    if strfind(scoredframes_list(sf).name, '.mat') >= 1
        % open the scoredframes file (loads as variable 'scored')
        load(fullfile(path_folderwithscores, scoredframes_list(sf).name))
        
        % open the scores file (loads as variable 'frames_to_score')
        load(fullfile(path_folderwithscores, scores_list(sf).name))
        
        % find the path for the images:
        % - find the index where the word 'seed' starts. omit the remainder
        % of the string after 'seed' to get the prefix for the cnn folder
        ind_seed = strfind(scoredframes_list(sf).name, 'seed');
        cnnimages_pre = scoredframes_list(1).name(1:(ind_seed-1));
        foldername = [cnnimages_pre 'frames_cnn'];
        folder_images = fullfile(path_cnnimages, foldername);
        
        % For each frame in this .mat file that is scored, copy the CNN
        % image into either 'freeze' or 'nofreeze' folder
        for frame = 1:size(scored, 1)
            filename = [cnnimages_pre 'frames_' num2str(frames_to_score(frame),'%05d') '.png'];
            source = fullfile(folder_images, filename);
            if scored(frame) == 1
                % if score is freeze, copy CNN image to freeze folder
                destination = fullfile(path_outputfolder, cnn_name,'nofreeze', filename);
                copyfile(source, destination)
                
            elseif scored(frame) == 0
                % if score is nofreeze, copy CNN image to nofreeze folder
                destination = fullfile(path_outputfolder, cnn_name,'freeze', filename);
                copyfile(source, destination)
            else
                % if score is NaN or something else, skip
            end
            
        end
    end
    
end

end

