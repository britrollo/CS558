function main ()
    %k-means
    k = 10;
    img = imread("white-tower.png");
    figure(1);
    imshow(img);
    result = k_means(k, img);
    figure(2);
    imshow(uint8(result)); %why is it not printing color?
   
    %SLIC 
%     img2 = imread("wt_slic.png");
%     figure(3);
%     imshow(img2);
    
end

function result = point_check(lst, rgb)
    % Parameters
        % lst -> matrix of rgb values
        % rgb -> rgb values to check
    % Return 
        % result -> 
            % 1 -> if x,y is NOT in lst
            % 0 -> if x,y is in lst
    
    l = size(lst);
    result = 1;
    r = rgb(1);
    g = rgb(2);
    b = rgb(3);
    for i = 1:l
        if lst(i, 1) == r && lst(i, 2) == g && lst(i, 3) == b
            result = 0;
            return;
        end
    end
    return;
end

function result = closest_cluster(img, px, py, c)
    % Parameters
        % img -> original image
        % px,py -> point
        % cx,cy -> cluster center
    % Return
        % result -> i of closest cluster ci
    closest_dist = inf;
    closest_c = 0;
    k = length(c);
    % Point r, g, b values
    prgb = img(px, py, :);
    pr = prgb(1);
    pg = prgb(2);
    pb = prgb(3);
    for i = 1:k
        % Cluster RGB values 
        crgb = c(i, :);
        cr = crgb(1);
        cg = crgb(2);
        cb = crgb(3);
        
        % Calculate color distance
        r = (cr - pr)^2;
        g = (cg - pg)^2;
        b = (cb - pb)^2;
        dis = sqrt(r + g + b);
        if dis < closest_dist
            closest_dist = dis;
            closest_c = i;
        end
    end
    result = closest_c;
    return;
end

function result = check_ci(og, newc, k)
    % Parameters
        % og -> original cluster centers
        % newc -> new cluster centers
    % Return
        % result ->
            % 1 -> original and new cluster centers are the same
            % 0 -> original and new cluster centers are NOT the same
    og = sortrows(og);
    newc = sortrows(newc);
    result = isequal(og, newc);
    return;
end

function result = k_means(k, img)
    img = double(img);

    %Get size of image
    [X, Y, col] = size(img);
    
%%%%% STEP 1: Randomly initialize the cluster centers, c1, ..., ck

    % Initialize cluster centers
    cluster_centers = zeros(k, 3);
    
    % Randomly pick k clusters and get rgb values
    for c = 1:k
        px = randi([1, X], 1, 1);
        py = randi([1, Y], 1, 1);

        crgb = img(px, py, :);
        
        
        % Check that the cluster center has not already been chosen
        while point_check(cluster_centers, crgb) == 0
            px = randi([1, X], 1, 1);
            py = randi([1, Y], 1, 1);
            crgb = img(px, py, :);
        end
        cluster_centers(c, 1) = crgb(1);
        cluster_centers(c, 2) = crgb(2);
        cluster_centers(c, 3) = crgb(3);
    end
    
    flag = 0;
    while flag == 0
%%%%%%%% STEP 2: Given cluster centers, determine points in each cluster
%%%%%%%%         For each point p, find the closest ci. Put p into cluster i
        
        
        % initialize matrix for holding points per cluster
        % X*Y points, [x, y, ci]
        clustered = zeros(X*Y, 3);
        cnt = 1;
        for x=1:X
            for y=1:Y
                % Find closest cluster center
                ci = closest_cluster(img, x, y, cluster_centers);
                clustered(cnt, 3) = ci;
                clustered(cnt, 1) = x;
                clustered(cnt, 2) = y;
                cnt = cnt + 1;
            end
        end
%%%%%%%% STEP 3: Given points in each cluster, solve for ci
%%%%%%%%         Set ci to be the mean of points in cluster i
        new_cluster_centers = zeros(k, 3);
        for ci=1:k
            n = 0;
            sumr = 0;
            sumg = 0;
            sumb = 0;
            for i=1:(X*Y)
                if clustered(i, 3) == ci
                    n = n + 1;
                    % pixel coordinates from cluster ci
                    px = clustered(i,1);
                    py = clustered(i,2);
                    % RGB value from original image
                    rgb = img(px, py, :);
                    sumr = sumr + rgb(1);
                    sumg = sumg + rgb(2);
                    sumb = sumb + rgb(3);
                end
            end
            new_cluster_centers(ci, 1) = floor(sumr/n);
            new_cluster_centers(ci, 2) = floor(sumg/n);
            new_cluster_centers(ci, 3) = floor(sumb/n); 
        end

%%%%%%%% STEP 4: If ci have changed, repeat STEP 2
        flag = check_ci(cluster_centers, new_cluster_centers, k);
        if flag == 0
            % Check for NaN values and 
            % keep original center is new one is NaN
            
%             new_cluster_centers = remove_NaN(cluster_centers, new_cluster_centers, k);
            cluster_centers = new_cluster_centers;
        end
    end
    
    % Create final image - Represent each cluster with the average RGB
    % value of its members
    
    % Initialize resulting image
    c_img = zeros(X, Y, 3);
    % Calculate average RGB value of cluster
    c_avg_rgb = zeros(k, 3);
    for ci=1:k
        n = 0;
        sumr = 0;
        sumg = 0;
        sumb = 0;
        for i=1:(X*Y)
            if clustered(i, 3) == ci
                n = n + 1;
                % pixel coordinates from cluster ci
                px = clustered(i,1);
                py = clustered(i,2);
                % RGB value from original image
                rgb = img(px, py, :);
                sumr = sumr + rgb(1);
                sumg = sumg + rgb(2);
                sumb = sumb + rgb(3);
            end
            c_avg_rgb(ci, 1) = floor(sumr/n);
            c_avg_rgb(ci, 2) = floor(sumg/n);
            c_avg_rgb(ci, 3) = floor(sumb/n);
        end
    end
    
    % Set new image coordinates to RGB average of each cluster 
    for ci=1:k
        for i=1:(X*Y)
            if clustered(i, 3) == ci
                x = clustered(i, 1);
                y = clustered(i, 2);
                r = c_avg_rgb(ci, 1);
                g = c_avg_rgb(ci, 2);
                b = c_avg_rgb(ci, 3);
                c_img(x, y, 1) = r;
                c_img(x, y, 2) = g;
                c_img(x, y, 3) = b;
            end
        end
    end
    
    result = c_img;  
end 


function result = slic(img)
%     get size of image
    [X, Y, c] = size(img);
    
%     Set maximum iteration value
    max_iter = 3;
    
%%%%% STEP 1: Initialization: Divide the image in blocks of
%%%%% 50x50 pixels

%%%%% initialize a centroid at the center of each block

%%%%% STEP 2: Local Shift: Compute the magnitude of the 
%%%%% gradient in each of the RGB channels 

%%%%% use the square root of the sum of squares of the three 
%%%%% magnitudes as the combined gradient magnitude

%%%%% Move the centroids to the position with the smallest 
%%%%% gradient magnitude in 3x3 windows centered on the 
%%%%% initial centroids.

%%%%% STEP 3: Centroid Update: Assign each pixel to its 
%%%%% nearest centroid in the 5D space of x, y, R, G, B
%%%%% and recompute centroids. Use the Euclidean distance
%%%%% in this space, but divide x and y by 2.

%%%%% STEP 4: Optionally: only compare pixels to centroids
%%%%% within a distance of 71 pixels (~sqrt(2)*50 block size)
%%%%% during the updates.

%%%%% STEP 5: If (not converged) and (iterations < max_iter)
%%%%% THEN go to STEP 2. max_iter = 3

%%%%% STEP 6: Display the output image as in teh SLIC slide:
%%%%% colorpixels that touch two different clusters black
%%%%% and the remaining pixels by the average RGB value of
%%%%% their cluster.
end