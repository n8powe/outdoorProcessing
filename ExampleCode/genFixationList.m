function fixation_list = genFixationList(fixation_frames)

%% use bool list to generate list of fixation blocks

%% indeces of fixation frames

frame_search = find(fixation_frames);

%% jumps in the index list are saccades

jumps = diff(frame_search); % gaps in fixation periods (saccades)
block_dex_end = frame_search(find(jumps>1)); % find gaps that are >1
block_dex_start = frame_search(find(jumps>1)+1); % indeces at where gaps are >1 are ends of fixation block, next index is where the next fixation block starts

%% generate fixation list (N x 2) where each row is the first and last frame of the nth of N fixations
for itr = 1:length(block_dex_start)-1

    fixation_list(itr,:) = [block_dex_start(itr) block_dex_end(itr+1)];

end




end