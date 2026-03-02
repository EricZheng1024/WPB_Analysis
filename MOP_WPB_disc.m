classdef MOP_WPB_disc < PROBLEM
% <multi/many> <real> <large/none> <expensive/none>
% MOP with the WPB and the disconnected PF
% d --- 0.9 --- Scalar or vector; i-th entry is for i-th objective; e.g. [0.7;0.9;0.8] for 3-objective case.
% r --- 0.1 --- Scalar or vector; i-th entry is for i-th objective; e.g. [0.1;0.2;0.3] for 3-objective case.
% ide ---  --- Scalar or vector; i-th entry is for i-th objective; e.g. [1;2;3] for 3-objective case.
% F --- 1 --- Scalar or vector; i-th entry is for i-th objective; e.g. [0.5;2;0.8] for 3-objective case.
% scaleF ---  --- Scalar or vector; e.g. [2;1] for the 2-objective case
% scalePF ---  --- Setting this parameter is the same as setting "scaleF". Scalar or vector; e.g. [2;1] for the 2-objective case

% Paper: Weak Pareto Boundary: A Critical Challenge for Evolutionary Multi-Objective Optimization
% Author: Ruihao Zheng

    properties(Access = private)
        d;
        r;
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
            [obj.d, obj.r, obj.ide, obj.F, obj.scaleF, obj.scalePF] = obj.ParameterSet(0.9, 0.1, (1:obj.M)', 1, ones(obj.M,1), ones(obj.M,1));
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
            if length(obj.r) <= 1
                obj.r = ones(obj.M,1)*obj.r;
            end
            if any(obj.r < 0)
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
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            [N,D]  = size(PopDec);
            M      = obj.M;
            g = zeros(N,M);
            for j = 1 : M
                X = PopDec(:, M+j:M:D);
                g(:,j) = 4*sum(2*abs(X - 0.5),2)/size(X,2);  % l is fixed
                % g(:,j) = 4*sum(4*(X - 0.5).^2,2)/size(X,2);
                % g(:,j) = 0;
            end

            h = PopDec(:,1:M) ./ sum(PopDec(:,1:M),2);
            index_nan = any(isnan(h),2);
            h(index_nan,:) = 1/M;

            h = obj.extend_h_exc_d(h)./(1+obj.r');
            h = h.^(obj.F');

            % h = (obj.scalePF').*(h.^(obj.F'));
            h = (obj.scalePF').*h;
            PopObj = (obj.scaleF').*(1+g).*h + obj.ide';
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            % R = obj.GetFeasibleLowerBound(N);
            R = obj.GetFeasibleLowerBound(min(1e6,max(N,nchoosek(50+obj.M-1,obj.M-1))));  % nchoosek(H+M-1,M-1), H is the number of sampling points in each dimension
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
                    tmp = obj.GetFeasibleLowerBound(N_sqrt^2,true);  % Uniform sampling in PS becomes non-uniform in PF; will definitely generate N_sqrt^2 sampling points
                    % tmp(NDSort(tmp,1)~=1) = nan;

                    % index = any(abs(tmp - (obj.d)') < 0.025, 2);
                    % index = any(tmp - (obj.d)' > -0.025 & tmp - (obj.d)' < 0, 2);
                    % index = any(tmp > (-0.025+(obj.d)').*(obj.scaleF'.*obj.scalePF') & tmp - (obj.d)'.*(obj.scaleF'.*obj.scalePF') < 0, 2);
                    % index = any(tmp-obj.ide' > (-0.025+(obj.d)').*(obj.scaleF'.*obj.scalePF') & tmp-obj.ide' - (obj.d)'.*(obj.scaleF'.*obj.scalePF') < 0, 2);  % To adapt to F settings, put in GetFeasibleLowerBound
                    % tmp(index,:) = nan;
                    
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
        function [R,N,index] = GetFeasibleLowerBound(obj,N,type)
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
                R = UniformPoint(N,obj.M);
                R(R==1e-6) = 0;
            end

            % R = obj.extend_h_exc_d(R);
            % if type  % for plot
            %     index = any(R - (obj.d)' > -0.025 & R - (obj.d)' < 0, 2);
            %     R(index,:) = nan;
            % end
            R = obj.extend_h_exc_d(R)./(1+obj.r');
            if type  % for plot
                index = any(R - (obj.d')./(1+obj.r') > -0.025 & R - (obj.d')./(1+obj.r') < 0, 2);  % Missing part in the middle of PF
                % index = any(R - (obj.d')./(1+obj.r') > 0 & R - (obj.d')./(1+obj.r') < 0.05, 2);  % Missing part at PF boundary, easily fails when PF is convex
                R(index,:) = nan;
            end
            R = R.^(obj.F');
            % R = (obj.scaleF').*(obj.scalePF').*(R.^(obj.F'));
            R = (obj.scaleF').*(obj.scalePF').*R + obj.ide';
        end
        %% 
        function h = extend_h_exc_d(obj, h)
            % index = h > (obj.d)';
            % tmp = sum(index,2);
            % for i = 1 : size(h,1)
            %     if tmp(i) < 1
            %         continue
            %     % elseif tmp > 1
            %     %     if tmp - index(i,end) > 1
            %     %         error('Parameter d is too small.');
            %     %     end
            %     %     index(i,end) = false;
            %     end
            %     h(i, index(i,:)) = h(i, index(i,:)) + obj.r(index(i,:));
            %     % h(i, ~index(i,:)) = h(i, ~index(i,:))*1.2;  % Convenient for plotting, HDB will not be drawn; but will make different segments of PF non-parallel
            %     % h(i,:) = h(i,:)*1.2;  % Convenient for plotting, HDB will not be drawn
            % end

            h = h + obj.r' .* (h > (obj.d)');
        end
    end
end