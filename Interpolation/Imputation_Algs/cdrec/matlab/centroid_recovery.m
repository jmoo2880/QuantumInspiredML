% [MATLAB API function]
% Centroid Recovery - uses truncated CD to impute all missing values in the matrix (designated with NaN)
function Xrec = centroid_recovery(matrix, varargin)% var = truncation(0), maxIterations(100), threshold(1E-6)
    % input processing
    if nargin > 1
        truncation = varargin{1};
    else
        truncation = 0;
    end
    
    if nargin > 2
        maxIterations = varargin{2};
    else
        maxIterations = 100;
    end
    
    if nargin > 3
        threshold = varargin{3};
    else
        threshold = 1E-6;
    end
    
    if nargin > 4
        disp("[Centroid Recovery] Warning: more than 4 arguments provided during the call. Anything beyond 4 is ignored.");
    end
    
    [n, m] = size(matrix);
    
    if truncation > m %strictly bigger
        disp("[Centroid Recovery] Error: provided truncation parameter k=" + tostring(truncation) + " is larger than the number of columns m=" + tostring(m));
        disp("[Centroid Recovery] Aborting recovery. Please provide a valid truncation parameter 1 <= k <= m - 1.");
        disp("[Centroid Recovery] Alternatively, providing k = 0 or k = m will choose one automatically.");
        return; % returns not initialized
    end
    
    if truncation == 0 || truncation == m
        truncation = 3;
    end
    
    truncation = min(truncation, m - 1);
    truncation = max(truncation, 1);
    maxIterations = max(maxIterations, 1);
    
    miss_mask = isnan(matrix);
    miss_count = sum(sum(miss_mask));
    
    if miss_count == 0
        disp("[Centroid Recovery] Warning: provided matrix doesn't contain any missing values.");
        disp("[Centroid Recovery] The algorithm will run, but will return an unchanged matrix.");
    end
    
    % initiate missing values
    matrix = interpolate(matrix);
    
    % init persistent values
    SV = default_SV(n, truncation);
    iter = 0;
    last_diff = threshold + 1.0; %dummy to ensure it doesn't terminate in 1 hop
    
    % main loop
    while iter < maxIterations && last_diff >= threshold
        % terminated if we reach the interation cap
        % or if our change to missing values from last iteration is small enough
        iter = iter+ 1;
        
        % perform truncated decomposition
        [L, R, SV] = centroid_decomposition(matrix, truncation, SV);
        
        % perform a low-rank reconstruction of the original matrix
        recon = L * R';
        
        % compute how much did it change using ||X[mis] - Xrec[mis]||_F / sqrt(|mis|)
        diff_vector = matrix(miss_mask) - recon(miss_mask);
        last_diff = norm(diff_vector) / sqrt(miss_count);
        
        % substitute values in the missing blocks with what was reconstructed after truncated CD
        matrix(miss_mask) = recon(miss_mask);
    end
    
    Xrec = matrix;
end


% simple linear interpolation function
% interpolates segments which are marked as NaN
% if the segments start (or ends) at the start (or end) of the column - uses 1NN instead
function matrix_ret = interpolate(matrix)
    [n, m] = size(matrix);
    
    for j = 1:m
        mb_start = -1;
        prev_value = NaN;
        step = 0; %init
        
        for i = 1:n
            if isnan(matrix(i, j))
                % current value is missing - we either start a new block, or we are in the middle of one

                if mb_start == -1
                    % new missing block
                    mb_start = i;
                    mb_end = mb_start + 1;
                    
                    while (mb_end <= n) && isnan(matrix(mb_end, j))
                        mb_end = mb_end+ 1;
                    end
                    
                    if mb_end == n+1
                        next_value = NaN;
                    else
                        next_value = matrix(mb_end, j);
                    end

                    if mb_start == 1 % special case %1: block starts with array
                        prev_value = next_value;
                    end
                    
                    if mb_end == n+1 % special case %2: block ends with array
                        next_value = prev_value;
                    end
                    
                    step = (next_value - prev_value) / (mb_end - mb_start + 1);
                end
                
                matrix(i, j) = prev_value + step * (i - mb_start + 1);
            else
                % missing block either ended just new or we're traversing normal data
                prev_value = matrix(i, j);
                mb_start = -1;
            end
        end
    end
    
    matrix_ret = matrix;
end


%%
%% decomposition functions
%%

% Centroid Decomposition, with the optional possibility of specifying truncation or usage of initial sign vectors
function [L_ret, R_ret, varargout] = centroid_decomposition(matrix, varargin)% var = truncation(0), SV([])
    % input processing
    if nargin > 1
        truncation = varargin{1};
    else
        truncation = 0;
    end
    
    if nargin > 2
        SV = varargin{2};
    else
        SV = [];
    end
    
    if nargin > 3
        disp("[Centroid Decomposition] Warning: more than 3 arguments provided during the call. Anything beyond 3 is ignored.");
    end
    
    [n, m] = size(matrix);
    
    if truncation == 0
        truncation = m;
    end
    
    if truncation < 1 || truncation > m
        disp("[Centroid Decomposition] Error: invalid truncation parameter k=" + tostring(truncation));
        disp("[Centroid Decomposition] Aboritng decomposition");
        return; % returns not initialized
    end
    
    if size(SV, 1) == 0
        SV = default_SV(n, truncation);
    end
    
    if size(SV, 2) ~= truncation
        disp("[Centroid Decomposition] Error: provided list of Sign Vectors doesn't match in size with the truncation truncation parameter k=" + tostring(truncation));
        disp("[Centroid Decomposition] Aboritng decomposition");
        return; % returns not initialized
    end
    
    L = zeros(truncation, n);
    R = zeros(truncation, m);
    
    % main loop - goes up till the truncation param (maximum of which is the % of columns)
    for j = 1:truncation
        % calculate the sign vector
        Z = local_sign_vector(matrix, SV(:, j));
        
        % calculate the column of R by X^T * Z / ||X^T * Z||
        R_i = matrix' * Z;
        R_i = R_i / norm(R_i);
        R(j,:) = R_i;
        
        % calculate the column of L by X * R_i
        L_i = matrix * R_i;
        L(j,:) = L_i;
        
        % subtract the dimension generated by L_i and R_i from the original matrix
        matrix = matrix - L_i * R_i';
        
        % update the new sign vector in the array
        SV(:, j) = Z;
    end
    
    L_ret = L';
    R_ret = R';
    if nargout > 1
        varargout{1} = SV;
    end
end


% Algorithm: LSV (Local Sign Vector). Finds locally optimal sign vector Z, i.e.:
%   Z being locally optimal means: for all Z' sign vectors s.t. Z' is one sign flip away from Z at some index j,
%   we have that ||X^T * Z|| >= ||X^T * Z'||
function Z_ret = local_sign_vector(matrix, Z)
    [n, m] = size(matrix);
    
    Z = local_sign_vector_init(matrix, Z);
    
    % calculate initial product of X^T * Z with the current version of Z
    direction = matrix' * Z;
    % calculate initial value of ||X^T * Z||
    lastNorm = norm(direction) ^ 2 + eps;
    
    flipped = true;
    
    while flipped
        % we terminate the loop if during the last pass we didn't flip a single sign
        flipped = false;
        
        for i = 1:n
            signDouble = Z(i) * 2;
            gradFlip = 0.0;
            
            % calculate how ||X^T * Z|| would change if we would change the sign at position i
            % change to the values of D = X^T * Z is calculated as D_j_new = D_j - 2 * Z_i * M_ij for all j
            gradFlip = norm(direction - signDouble * matrix(i, :));
            
            % if it results in augmenting ||X^T * Z||
            % flip the sign and replace cached version of X^T * Z and its norm
            if gradFlip > lastNorm
                flipped = true;
                Z(i) = Z(i) * -1;
                lastNorm = gradFlip + eps;
                
                direction = direction - signDouble * matrix(i, :);
            end
        end
    end
    
    Z_ret = Z;
end


% Auxiliary function for LSV:
%   Z is initialized sequentiually where at each step we see which sign would give a larger increase to ||X^T * Z||
function Z_ret = local_sign_vector_init(matrix, Z)
    [n, m] = size(matrix);
    
    direction = matrix(1, :);
    
    for i = 2:n
        gradPlus = norm(direction + matrix(i, :));
        gradMinus = norm(direction - matrix(i, :));
        
        if gradMinus > gradPlus
            Z(i) = -1;
        end
        
        direction = direction + Z(i) * matrix(i, :);
    end
    
    Z_ret = Z;
end


%initialize sign vector array with default values
function SV = default_SV(n, k)
    % default sign vector is (1, 1, ..., 1)^T
    SV = ones(n, k);
end