function main ()
    %k-means
    k = 10;
    img = imread("white-tower.png");
    
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

function result = k_means(k, img)
    %Get size of image
    [X, Y] = size(img);
    
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
        end
    end

%%% STEP 3: Given points in each cluster, solve for ci
%%%         Set ci to be the mean of points in cluster i
    

%%% STEP 4: If ci have changed, repeat STEP 2
    
end 