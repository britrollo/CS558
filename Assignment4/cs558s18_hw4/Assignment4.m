function main()
    % Problem 1 : Image Classification
    bins = 8;
    directory = "ImClass";
    classifyIm(bins, directory); 
    % See if you can increase the accuracy by changing the number of bins in the histogram 
    bins2 = 16;
    classifyIm(bins2, directory);
    
    % Problem 2: Pixel Classification
    k = 10;
    directory = "sky";
    classifyPix(k, directory);
    
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
function classifyIm(bins, directory)
% Step1: Get images from ImClass directory
    test_images = dir(fullfile(directory, "*test*.jpg"));
    train_images = dir(fullfile(directory, "*train*.jpg"));
% Step2: For training set images, Create 3 separate histograms (with 8 bins each) of the R, G, and B color channels
% each image is represented by 24 numbers
    train_hists(1:length(train_images)) = struct('r', zeros(1, bins), 'g', zeros(1, bins), 'b', zeros(1, bins));
    vote_check = 0;
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
        if result ~= X*Y*3
            disp("Voting Error: " + train_images(ii).name);
        else
            vote_check = vote_check + 1;
        end
    end
    if vote_check == length(train_images)
        disp("Each pixel of all training images has 3 votes.");
    end
    
    % Step 5: compute 3 histogram representation for test images
    test_hists(1:length(test_images)) = struct('r', zeros(1, bins), 'g', zeros(1, bins), 'b', zeros(1, bins));
    correct = 0;
    vote_check = 0;
    for ii=1:length(test_images)
        img = imread(strcat(test_images(ii).folder, "/", test_images(ii).name));
        img = double(img);
        [X, Y, ~] = size(img);
        for i=1:X
            for j=1:Y
                % Step6: Each pixel votes in each histogram
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
        result = sum(test_hists(ii).r) + sum(test_hists(ii).g) + sum(test_hists(ii).b);
        if X*Y*3 == result
            vote_check = vote_check + 1;
        else
           disp("Voting Error: " + trest_images(ii).name);
        end
        
        % Step7: Assign to the test image the label of the training image
        % that has the "nearest" representation
        % "nearest" representation computed by using the Euclidean distance
        % in the 24D histogram space 
        closest = twentyfourdistance(test_hists(ii), train_hists);
        disp("Test image " + string(ii) + " of class " + string(ceil(ii/4)) + " has been assigned to class " + string(ceil(closest/4)) + ".");
        if ceil(ii/4) == ceil(closest/4)
            correct = correct + 1;
        end
    end
    if vote_check == length(test_images)
        disp("Each pixel of all test images has 3 votes.");
    end
    % Step8: Compute accuracy of classifier
    disp("Bins: " + string(bins) + " -- Accuracy: " + string(correct/length(test_images)));
end

% PROBLEM 2: Pixel Classification
function classifyPix(k, directory)
    % Step1: Load both the original input image and the one you created which is
    % used as to guide the formation of the training set
    train_images = dir(fullfile(directory, "*train*.jpg"));
    test_images = dir(fullfile(directory, "*test*.jpg"));
    non = dir(fullfile(directory, "non*train*.jpg"));
    for ii=1:length(train_images)
        if strcmp(non.name, train_images(ii).name) == 0
            img = imread(strcat(train_images(ii).folder, "/", train_images(ii).name));
        else
            mask = imread(strcat(train_images(ii).folder, "/", train_images(ii).name));
        end
    end
    %Step2: Use nonsky as a mask to separate sky from non-sky pixels during
    %training
    [X, Y, ~] = size(img);
    sky = [];
    nonsky = [];
    sky_idx = 1;
    nonsky_idx = 1;
    for i=1:X
        for j=1:Y
            if mask(i,j,1) == 255 && mask(i,j,2) == 255 && mask(i,j,3) == 255
                sky(sky_idx, :) = img(i,j,:);
                sky_idx = sky_idx + 1;
            else
                nonsky(nonsky_idx, :) = img(i,j,:);
                nonsky_idx = nonsky_idx + 1;
            end
        end
    end
    
    %Step3: Run k-means separately on the sky and non-sky sets with k=10 to
    % obtain 10 visual words for each class
    % kmeans(sky, k, 'EmptyAction, 'singleton')
    [~, sky_word] = kmeans(sky, k, 'EmptyAction', 'singleton');     %10x3
    [~, nonsky_word] = kmeans(nonsky, k, 'EmptyAction', 'singleton'); 
    word=[ones(k,1) sky_word;zeros(k,1) nonsky_word]; % combine all words into one word list, marking sky 1 and nonsky 0
    
    %Step4: for each pixel of the test image find the nearest word and
    %classify it as sky or nonsky (brute force acceptable)
    for i=1:length(test_images)
        img = imread(strcat(test_images(i).folder, "/", test_images(i).name));
        [X, Y, c] = size(img);
        reshape_img = double(reshape(img, X*Y, c, 1)); %Reshape image's matrix
        
        word_idx = knnsearch(word(:,2:end),reshape_img,'k',1,'Distance','euclidean');
        closest_word = word(word_idx,1);
        [x,y] = ind2sub([X Y], 1:X*Y);
        
        for j=1:X*Y
            %Step5: Generate an output image in which the sky pixels are painted
            %with a distinctive color
            if closest_word(j) == 1
                img(x(j),y(j),1)=255;
                img(x(j),y(j),2)=0;
                img(x(j),y(j),3)=0;
            end
        end
        figure(i);
        imshow(img);
    end
end
