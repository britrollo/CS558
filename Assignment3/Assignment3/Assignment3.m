function main ()
    %k-means
    k = 10;
    img = imread("white-tower.png");
    result = k_means(k, img)
    imshow(result);
    %SLIC 
    
end

function result = point_check(lst, x, y)
    % Parameters
        % lst -> matrix of points
        % x,y -> point to check
    % Return 
        % result -> 
            % true -> if x,y is NOT in lst
            % false -> if x,y is in lst
    
    l = size(lst);
    result = true;
    for i = 1:l
        if lst(i, 1) == x && lst(i, 2) == y
            result = false;
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
        % result -> number of closest cluster
    closest_dist = inf;
    closest_c = 0;
    k = length(c);
    for i = 1:k
        % Cluster x,y 
        cxy = c(k, :);
        cx = cxy(1);
        cy = cxy(2);
        % Cluster r, g, b values
        crgb = img(cx, cy, :);
        cr = crgb(1);
        cg = crgb(2);
        cb = crgb(3);
        
        % Point r, g, b values
        prgb = img(px, py, :);
        pr = prgb(1);
        pg = prgb(2);
        pb = prgb(3);
        
        % Calculate color distance
        dis = sqrt((cr - pr)^2 + (cg - pg)^2 + (cb - pb)^2);
        if dis < closest_dist
            closest_dist = dist;
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
            % true -> original and new cluster centers are the same
            % false -> original and new cluster centers are NOT the same
    found = zeros(1,k);
    for i=1:k
        for j=1:k
            if (og(i,1) == newc(j,1)) && (og(j,2) == newc(j,2))
                found(i) = 1;
                break;
            end
        end
    end
    if sum(found) == k
        result = true;
        return;
    else
        result = false;
        return;
    end
end

function result = k_means(k, img)
    %Get size of image
    [X, Y, col] = size(img);
    
%%% STEP 1: Randomly initialize the cluster centers, c1, ..., ck

    % Initialize cluster centers
    cluster_centers = zeros(k, 2);
    
    % Randomly pick k clusters
    for c = 1:k
        px = randi([1, X], 1, 1);
        py = randi([1, Y], 1, 1);
        
        % Check that the cluster center has not already been chosen
        while point_check(cluster_centers, px, py) == false
            px = randi([1, X], 1, 1);
            py = randi([1, Y], 1, 1);
        end
        cluster_centers(c, 1) = px;
        cluster_centers(c, 2) = py;
    end
    
    flag = false;
    while flag ~= true
    %%% STEP 2: Given cluster centers, determine points in each cluster
    %%%         For each point p, find the closest ci. Put p into cluster i

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

    %%% STEP 3: Given points in each cluster, solve for ci
    %%%         Set ci to be the mean of points in cluster i
        new_cluster_centers = zeros(k, 2);
        for ci=1:k
            n = 0;
            sumx = 0;
            sumy = 0;
            for i=1:(X*Y)
                if clustered(i, 3) == ci
                    n = n + 1;
                    sumx = sumx + clustered(i, 1);
                    sumy = sumy + clustered(i, 2);
                end
            end
            new_cluster_centers(ci, 1) = floor(sumx/n);
            new_cluster_centers(ci, 2) = floor(sumy/n);
        end

    %%% STEP 4: If ci have changed, repeat STEP 2
        flag = check_ci(cluster_centers, new_cluster_centers, k);
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