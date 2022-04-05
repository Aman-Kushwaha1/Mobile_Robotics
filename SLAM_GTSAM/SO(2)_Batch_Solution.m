
import gtsam.*
path = 'C:\Users\a4ama\Desktop\gtsam\input_INTEL_g2o.g2o';
fileID = fopen(path,'r');

data = textscan(fileID,'%s%f%f%f%f%f%f%f%f%f%f%f');

pose = []; edge = [];
for j=1:size(data{1})
       if data{1}(j)=="VERTEX_SE2"
           pose = [pose;data{3}(j) data{4}(j) data{5}(j)];   %data{2}(j);
       elseif data{1}(j)=="EDGE_SE2"
           edge = [edge;data{2}(j) data{3}(j)  data{4}(j) data{5}(j) data{6}(j) data{7}(j) data{8}(j) data{9}(j)...
                    data{10}(j) data{11}(j) data{12}(j)];
       end
end

%creating 2d non-linear factorgraph
graph = NonlinearFactorGraph;   

%adding prior for the first location
priorModel = noiseModel.Diagonal.Sigmas([1.0, 1.0, 0.1]');
graph.add(PriorFactorPose2(symbol('x', 1), Pose2(0, 0, 0), priorModel));

odomModel = noiseModel.Diagonal.Sigmas([0.5, 0.5, 0.1]');

%add edges
for r = 1: size(edge)
    cov = [edge(r,6) edge(r,7) edge(r,8); edge(r,7) edge(r,9) edge(r,10); edge(r,8) edge(r,10) edge(r,11)];
    cov = inv(cov);
    odomModel = noiseModel.Gaussian.Covariance(cov);
    graph.add(BetweenFactorPose2(symbol('x', edge(r,1)+1), symbol('x', edge(r,2)+1), Pose2(edge(r,3), edge(r,4), edge(r,5)), odomModel));

end

initials = Values;
for r = 1:size(pose)
        initials.insert(symbol('x', r), Pose2(pose(r,1), pose(r,2), pose(r,3)));
end
  
% optimize!
optimizer = GaussNewtonOptimizer(graph, initials);
results = optimizer.optimizeSafely();

% plot result trajectory
figure(1)
plot(pose(:,1),pose(:,2))
hold on
plot2DTrajectory(results, 'r');
legend("Unoptimized trajectory", "Optimized trajectory")

