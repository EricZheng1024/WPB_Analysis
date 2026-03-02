classdef MOP_WPB_conn < PROBLEM
% <multi/many> <real> <large/none> <expensive/none>
% MOP with the WPB and the connected PF
% d --- 0.9 --- Scalar or vector; i-th entry is for i-th objective; e.g. [0.7;0.9;0.8] for 3-objective case.
% l --- 4 --- Scalar or vector; i-th entry is for i-th objective; e.g. [1;10;100] for 3-objective case.
% ide ---  --- Scalar or vector; i-th entry is for i-th objective; e.g. [1;2;3] for 3-objective case.
% F --- 1 --- Scalar or vector; i-th entry is for i-th objective; e.g. [0.5;2;0.8] for 3-objective case.
% scaleF ---  --- Scalar or vector; e.g. [2;1] for the 2-objective case
% scalePF ---  --- Setting this parameter is the same as setting "scaleF". Scalar or vector; e.g. [2;1] for the 2-objective case

% Paper: Weak Pareto Boundary: A Critical Challenge for Evolutionary Multi-Objective Optimization
% Author: Ruihao Zheng

    properties(Access = private)
        d;  % width
        l;  % length
        ide;
        F;
        scaleF;
        scalePF;
    end
    
    methods
        %% Default settings of the problem
        function Setting(obj)
            if isempty(obj.M); obj.M = 3; end
            if isempty(obj.D); obj.D = 2*obj.M; end
            [obj.d, obj.l, obj.ide, obj.F, obj.scaleF, obj.scalePF] = ...
                obj.ParameterSet(0.9, 4, (1:obj.M)', 1, ones(obj.M,1), ones(obj.M,1));
            % d
            if length(obj.d) <= 1
                obj.d = ones(obj.M,1)*obj.d;
            end
            if any(obj.d < 1/(obj.M-1))
                error('Parameter d is too small.');
            elseif any(obj.d > 1)
                warning('The setting of Parameter d is invalid.');
            end
            % l
            if length(obj.l) <= 1
                obj.l = ones(obj.M,1)*obj.l;
            end
            if any(obj.l < 0)
                error('Parameter l should be non-negative.');
            end
            % ide
            if length(obj.ide) <= 1
                obj.ide = ones(obj.M,1)*obj.ide;
            end
            % F
            if length(obj.F) <= 1
                obj.F = ones(obj.M,1)*obj.F;
            end
            if any(obj.F < 0)
                error('Parameter F should be non-negative.');
            end
            % scale
            if length(obj.scaleF) <= 1
                obj.scaleF = ones(obj.M,1)*obj.scaleF;
            end
            if any(obj.scaleF < 0)
                error('Parameter scaleF should be non-negative.');
            end
            if length(obj.scalePF) <= 1
                obj.scalePF = ones(obj.M,1)*obj.scalePF;
            end
            if any(obj.scalePF < 0)
                error('Parameter scalePF should be non-negative.');
            end
            % variable
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.encoding = 'real';
        end
        %% Get parameters
        function res = get(obj,str)
            switch str
                case 'd'
                    res = obj.d;
                case 'l'
                    res = obj.l;
                case 'ide'
                    res = obj.ide;
                case 'F'
                    res = obj.F;
                case 'scaleF'
                    res = obj.scaleF;
                case 'scalePF'
                    res = obj.scalePF;
                otherwise
                    res = [];
            end
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            [N,D]  = size(PopDec);
            M      = obj.M;

            g = zeros(N,M);
            for j = 1 : M
                X = PopDec(:, M+j:M:D);
                % g(:,j) = 4*sum(2*abs(X - 0.5),2)/size(X,2);
                % g(:,j) = 4*sum(4*(X - 0.5).^2,2)/size(X,2);
                % g(:,j) = 0;
                g(:,j) = obj.l(j)*sum(2*abs(X - 0.5),2)/size(X,2);
            end

            h = PopDec(:,1:M) ./ sum(PopDec(:,1:M),2);
            index_nan = any(isnan(h),2);
            h(index_nan,:) = 1/M;

            h = obj.project_h_exc_d(h);
            h = h ./ obj.d';  % Scale the nadir of PF to 1 so that the nadir remains unchanged after convexity/concavity transformation

            h = (obj.scalePF').*(h.^(obj.F'));

            % Variable linkage (after nonlinear transformation)
            % g = zeros(N,M);
            % for j = 1 : M
            %     X = PopDec(:, M+j:M:D);
            %     % g(:,j) = obj.l(j)*sum(2*abs(X - h(:,j)),2)./h(:,j)/size(X,2);  % h(:,j) can be 0 
            %     g(:,j) = obj.l(j)*sum(2*abs(X - h(:,j)),2)/size(X,2);
            % end

            PopObj = (obj.scaleF').*(1+g).*h + obj.ide';  % More easily guide the population close to HDB
            % PopObj = (obj.scaleF').*(g+h) + obj.ide';
        end
        %% Calculate constraint violations    <constrained/none>
        % function PopCon = CalCon(obj,PopDec)
        %     h = PopDec(:,1:obj.M) ./ sum(PopDec(:,1:obj.M),2);
        %     index_nan = any(isnan(h),2);
        %     h(index_nan,:) = 1/obj.M;
        %     PopCon = sum(max(h-(obj.d)',0), 2);
        % end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            % R = obj.GetFeasibleLowerBound(N);
            R = obj.GetFeasibleLowerBound(min(1e6,max(N,nchoosek(50+obj.M-1,obj.M-1))));  % nchoosek(H+M-1,M-1), H is the number of sampling points in each dimension
            % R = obj.GetFeasibleLowerBound(1000); warning('Test problem sampling points are temporarily set to few.')
            % R(NDSort(R,1)~=1,:) = [];
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            switch obj.M
                case 2
                    R = obj.GetFeasibleLowerBound(1000,true);
                    % R(NDSort(R,1)~=1) = nan;  % Since each row of R represents an objective vector, as long as each row has one nan, this notation is equivalent to R2(NDSort(R2,1)~=1,:) = nan; Same below
                case 3
                    N_sqrt = 40;
                    [tmp,~,exc_index,exc_cond_index] = obj.GetFeasibleLowerBound(N_sqrt^2,true);  % Uniform sampling in PS becomes non-uniform in PF; will definitely generate N_sqrt^2 sampling points
                    % nexc_index = setdiff(1:size(tmp,1), exc_index);  % Projected points may cause issues in non-dominated sorting due to computational errors, so only perform non-dominated sorting on non-projected points
                    % tmp(nexc_index(NDSort(tmp(nexc_index,:),1)~=1)) = nan;
                    tmp(exc_cond_index) = nan;
                    R = cell(1,3);
                    for i = 0 : (N_sqrt^2-1)
                        R{1}(mod(i,N_sqrt)+1,floor(i/N_sqrt)+1) = tmp(i+1,1);
                        R{2}(mod(i,N_sqrt)+1,floor(i/N_sqrt)+1) = tmp(i+1,2);
                        R{3}(mod(i,N_sqrt)+1,floor(i/N_sqrt)+1) = tmp(i+1,3);
                    end
                otherwise
                    R = [];
            end
        end
        %% Generate points on the lower boundary of feasible region
        function [R,N,exc_index,exc_cond_index] = GetFeasibleLowerBound(obj,N,type)
            switch nargin
                case 2
                    type = false;
                case 3
                    type = logical(type(1));
                otherwise
                    error('Too many input arguments.')
            end

            if type  % for plot
                [x_grid,N] = UniformPoint(N,obj.M-1,'grid');
                R = fliplr(cumprod([ones(N,1),x_grid(:,1:obj.M-1)],2)).*[ones(N,1),1-x_grid(:,obj.M-1:-1:1)];
            else
                % R = UniformPoint(N,obj.M);
                % disp('Test problem sampling point density is consistent')

                N0 = N;
                if N == nchoosek(50+obj.M-1,obj.M-1) && all(abs(obj.d - 1/(obj.M-1)) < 1e-6)
                    switch obj.M  % empirical value
                        case 4
                            N = 408426;
                        case 5
                            N = 8491251;
                    end
                end
                while true  % Eventually should generate more than N points
                    R = UniformPoint(N,obj.M);
                    % [N  N-sum(any(R>(obj.d)',2))  N0  N-sum(any(R>(obj.d)',2))-N0]
                    if N-sum(any(R>(obj.d)',2)) > N0
                        break
                    else
                        N = N + 5^obj.M;
                    end
                end
                % disp('Test problem sampling point count is consistent')

                R(R==1e-6) = 0;
            end

            % Record the latter half of points exceeding d, which can be used to draw a more aesthetically pleasing PF
            % if obj.M <= 3
                exc_index = R > (obj.d)';  % All indices exceeding d
                exc_cond_index = [];  % Indices exceeding d and meeting certain conditions
                % for i = 1 : size(R,1)
                %     tmp = sum(exc_index(i,:));
                %     if tmp < 1
                %         continue
                %     elseif tmp > 1
                %         if tmp - exc_index(i,end) > 1  % 2024.12.16 Can't remember the meaning of this "end"... might be for 3-objective case 
                %             error('Parameter d is too small.');
                %         end
                %         exc_index(i,end) = false;
                %     end
                % end
                for i = 1 : obj.M
                    index = find(exc_index(:,i));
                    tmp = sum(R(index,setdiff(1:obj.M,i)), 2);
                    exc_cond_index = [exc_cond_index index(tmp<=median(tmp))'];
                end
                exc_index = find(any(exc_index,2));
            % end

            % if obj.M <= 3  % inaccurate
            %     C = nchoosek(1:obj.M,obj.M-1);
            %     for i = 1 : size(C,1)
            %         tmp = sum(R(:,C(i,:)),2);
            %         index = tmp < (1-obj.d(setdiff(1:obj.M,C(i,:))));
            %         if any(index)
            %             max_tmp = maxk(tmp(index),ceil(sum(index)/2));
            %             exc_index = [exc_index find(tmp<=max_tmp(1))'];
            %             exc_cond_index = [exc_cond_index find(tmp<=max_tmp(end))'];
            %         end
            %     end
            % end

            if type
                R = obj.project_h_exc_d(R);
            else
                R(exc_index,:) = [];  % Do not perform mapping, otherwise an uneven optimal point set will be obtained
            end
            R = R ./ obj.d';  % Scale the nadir of PF to 1 so that the nadir remains unchanged after convexity/concavity transformation
            R = (obj.scaleF').*(obj.scalePF').*(R.^(obj.F')) + obj.ide';
        end
        %% Project points exceeding d in simplex h onto a line, making the position function a hexagon, which forms HDB when combined with distance function    flat fitness may occur
        % The idea is somewhat similar to "Effects of Dominance Resistant Solutions on the Performance of Evolutionary Multi-Objective and Many-Objective Algorithms", but mine is more elegant
        function h = project_h_exc_d(obj, h)
            % index = h > (obj.d)';
            % tmp = sum(index,2);
            % tmp_index = ~index;
            % for i = 1 : size(h,1)
            %     if tmp(i) < 1
            %         continue
            %     end
            %     h(i, tmp_index(i,:)) = h(i, tmp_index(i,:)) + ...
            %         sum(h(i, index(i,:)) - obj.d(index(i,:))')* ...
            %         (obj.d(tmp_index(i,:))'-h(i, tmp_index(i,:)))/sum(obj.d(tmp_index(i,:))'-h(i, tmp_index(i,:)));  % The larger the proportion, the smaller the addition
            %     h(i, index(i,:)) = obj.d(index(i,:));
            %     % if any(h(i,:) > (obj.d)')
            %     % end
            % end

            tmp = h - obj.d';
            h = min(h,obj.d') + ...
                sum(max(0, tmp), 2) .* ...
                max(0, -tmp) ./ ...
                sum(max(0, -tmp), 2);  % The larger the proportion, the smaller the addition
        end
    end
end

%% Note
% [x_grid,N] = UniformPoint(N,obj.M-1,'grid');
% R = fliplr(cumprod([ones(N,1),x_grid(:,1:obj.M-1)],2)).*[ones(N,1),1-x_grid(:,obj.M-1:-1:1)];
% Same as
% a = linspace(0,1,40)';
% R = {(a*a'), ...
%     (a*(1-a')), ...
%     ((1-a.*cos(obj.dc_A*a.^obj.dc_beta*pi).^2)*ones(size(a')))};
% Generate the same simplex