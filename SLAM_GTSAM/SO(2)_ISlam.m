
import gtsam.*
path = 'C:\Users\a4ama\Desktop\gtsam\input_INTEL_g2o.g2o';
fileID = fopen(path,'r');

data = textscan(fileID,'%s%f%f%f%f%f%f%f%f%f%f%f');

%Loading data
pose = []; edge = [];
for j=1:size(data{1})
       if data{1}(j)=="VERTEX_SE2"
           pose = [pose;data{3}(j) data{4}(j) data{5}(j)];   %data{2}(j);
       elseif data{1}(j)=="EDGE_SE2"
           edge = [edge;data{2}(j) data{3}(j)  data{4}(j) data{5}(j) data{6}(j) data{7}(j) data{8}(j) data{9}(j)...
                    data{10}(j) data{11}(j) data{12}(j)];
       end
end

%%%%%Q1 c

isam = gtsam.ISAM2();

for r = 1:size(pose)
    
    graph = NonlinearFactorGraph;
    initialEstimate = Values;
    
    if r==1
        priorModel = noiseModel.Diagonal.Sigmas([0, 0, 0]');
        graph.add(PriorFactorPose2(symbol('x', 1), Pose2(0, 0, 0), priorModel));
        
        %Adding initial estmate i.e. pose for x1
        initialEstimate.insert(symbol('x', 1), Pose2(pose(r,1), pose(r,2), pose(r,3)));
    else
        prevpose = result.at(symbol('x', r-1));
        initialEstimate.insert(symbol('x', r), Pose2(prevpose));
        for k= 1:size(edge)
            if (edge(k,2)+1) == r
                cov = [edge(k,6) edge(k,7) edge(k,8); edge(k,7) edge(k,9) edge(k,10); edge(k,8) edge(k,10) edge(k,11)];
                cov = inv(cov);
                model = noiseModel.Gaussian.Covariance(cov);
                graph.add(BetweenFactorPose2(symbol('x', 1+edge(k,1)), symbol('x', 1+edge(k,2)), Pose2(edge(k,3), edge(k,4), edge(k,5)), model));
               
              end
        end


    end
isam.update(graph,initialEstimate);
result = isam.calculateEstimate();

end 

figure(1)
plot(pose(:,1),pose(:,2))
hold on
plot2DTrajectory(result, 'r');
legend("Unoptimized trajectory", "Optimized trajectory")


