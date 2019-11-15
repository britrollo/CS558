function main()
% TODO
% run problem 1 with different amounts of bins
    bins = 8;
    directory = "ImClass";
    result = classifyIm(bins, directory);
end

% PROBLEM 1: Image Classification
function result = classifyIm(bins, directory)
% Step1: Get images from ImClass directory
    test_images = dir(fullfile(directory, "*test*.jpg"));
    train_images = dir(fullfile(directory, "*train*.jpg"));
% Step2: For training set images, Create 3 separate histograms (with 8 bins each) of the R, G, and B color channels
% each image is represented by 24 numbers
    train_hists(1:length(train_images)) = struct('r', zeros(1, bins), 'g', zeros(1, bins), 'b', zeros(1, bins));
    for ii=1:length(train_images)
        img = imread(strcat(train_images(ii).folder, "/", train_images(ii).name));
        img = double(img);
        [X, Y, ~] = size(img);
        for i=1:X
            for j=1:Y
                % Step3: Each pixel votes in each histogram
                div = ceil((255+1)/bins);   % (255+1) to cover range from 0 to 255
                rgb = img(i,j,:);
                rv = rgb(1)+1;              % +1 to adjust for 0
                train_hists(ii).r(ceil(rv/div)) = train_hists(ii).r(ceil(rv/div)) + 1;
                gv = rgb(2)+1;
                train_hists(ii).g(ceil(gv/div)) = train_hists(ii).g(ceil(gv/div)) + 1;
                bv = rgb(3)+1;
                train_hists(ii).b(ceil(bv/div)) = train_hists(ii).b(ceil(bv/div)) + 1;
            end
        end
        % Step4: Verify all pixels are counted exactly 3 times
        result = sum(train_hists(ii).r) + sum(train_hists(ii).g) + sum(train_hists(ii).b);
        if result == X*Y*3
            disp("Each pixel of " + train_images(ii).name + " has 3 votes.");
        else
            disp("Voting Error: " + train_images(ii).name);
        end
    end
    
% Classify each image into one of three classes: coast, forest, or "insidecity"
end
