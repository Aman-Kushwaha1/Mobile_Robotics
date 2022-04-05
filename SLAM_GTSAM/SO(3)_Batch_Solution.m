import gtsam.*
path = 'C:\Users\a4ama\Desktop\gtsam\parking-garage.g2o';
fileID = fopen(path,'r');

data = textscan(fileID,'%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f');

%Loading data
pose = []; edge = [];
for j=1:size(data{1})
       if data{1}(j)=="VERTEX_SE3:QUAT"
           pose = [pose; data{2}(j) data{3}(j) data{4}(j) data{5}(j) data{6}(j) data{7}(j) data{8}(j) data{9}(j)];
       elseif data{1}(j)=="EDGE_SE3:QUAT"
           edge = [edge; data{2}(j) data{3}(j) data{4}(j) data{5}(j) data{6}(j) data{7}(j) data{8}(j) data{9}(j)...
                    data{10}(j) data{11}(j) data{12}(j) data{13}(j) data{14}(j) data{15}(j) data{16}(j) data{17}(j) data{18}(j)...
                    data{19}(j) data{20}(j) data{21}(j) data{22}(j) data{23}(j) data{24}(j) data{25}(j) data{26}(j)...
                    data{27}(j) data{28}(j) data{29}(j) data{30}(j) data{31}(j)];
       end 
end

%homogeneous transformation matrix
htm = quat2tform([pose(1,8),pose(1,5),pose(1,6),pose(1,7)]);

%Using nonlinear FactorGraph
graph = NonlinearFactorGraph;
priorNoisecov = noiseModel.Diagonal.Sigmas([0.1; 0.1; 0.1; 0.1; 0.1; 0.1]);
graph.add(PriorFactorPose3(0, Pose3(htm), priorNoisecov));


for k=1:size(edge,1)
    info = eye(6);
    X = edge(k,25:30);
    info(4,4) = X(1,1);
    inf(4,5) = X(1,2);
    info(5,4) = X(1,2);
    info(4,6) = X(1,3);
    info(6,4) = X(1,3);
    info(5,5) = X(1,4);
    info(5,6) = X(1,5);
    info(6,5) = X(1,5);
    info(6,6) = X(1,6);
    covariance = inv(info);
    Model = noiseModel.Gaussian.Covariance(covariance);
    pose_new = quat2tform([edge(k,9),edge(k,6),edge(k,7),edge(k,8)]);
    pose_new(1:3,4) = edge(k,3:5)';
    graph.add(BetweenFactorPose3(edge(k,1), edge(k,2), Pose3(pose_new), Model));    
end


%Adding Initial Vlaues
initials = Values;
initials.insert(pose(1,1), Pose3(htm));
for i=2:size(pose,1)
    pose_new = quat2tform([pose(i,8),pose(i,5:7)]);
    pose_new(1:3,4) = pose(i,2:4)';
    initials.insert(pose(i,1), Pose3(pose_new)); 
end

optimizer = GaussNewtonOptimizer(graph, initials);
results = optimizer.optimizeSafely();

plot3DTrajectory(results,'b');
hold on
plot3DTrajectory(initials, 'r');

