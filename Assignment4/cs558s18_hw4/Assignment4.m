function main()
% TODO
% run problem 1 with different amounts of bins
    bins = 8;
    directory = "ImClass";
    result = classifyIm(bins, directory);
end

function result = twentyfourdistance(test, train)
    closest_dist = inf;
    idx = 0;
    for i=1:length(train)
        test_dist = sqrt((sum((test.r-train(i).r).^2))+(sum((test.g-train(i).g).^2))+(sum((test.b-train(i).b).^2)));
        if test_dist < closest_dist
            idx = i;
            closest_dist = test_dist;
        end
    end
    result = idx;
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
    
    % Step 5: compute 3 histogram representation for test images
    test_hists(1:length(test_images)) = struct('r', zeros(1, bins), 'g', zeros(1, bins), 'b', zeros(1, bins));
    correct = 0;
    for ii=1:length(test_images)
        img = imread(strcat(test_images(ii).folder, "/", test_images(ii).name));
        img = double(img);
        [X, Y, ~] = size(img);
        for i=1:X
            for j=1:Y
                % Step3: Each pixel votes in each histogram
                div = ceil((255+1)/bins);   % (255+1) to cover range from 0 to 255
                rgb = img(i,j,:);
                rv = rgb(1)+1;              % +1 to adjust for 0
                test_hists(ii).r(ceil(rv/div)) = test_hists(ii).r(ceil(rv/div)) + 1;
                gv = rgb(2)+1;
                test_hists(ii).g(ceil(gv/div)) = test_hists(ii).g(ceil(gv/div)) + 1;
                bv = rgb(3)+1;
                test_hists(ii).b(ceil(bv/div)) = test_hists(ii).b(ceil(bv/div)) + 1;
            end
        end
        closest = twentyfourdistance(test_hists(ii), train_hists);
        disp("Test image " + string(ii) + " of class " + string(ceil(ii/4)) + " has been assigned to class " + string(ceil(closest/4)) + ".");
        if ceil(ii/4) == ceil(closest/4)
            correct = correct + 1;
        end
    end
    disp("Accuracy: " + string(correct/length(test_images)));
    
% Classify each image into one of three classes: coast, forest, or "insidecity"
end
