% --------------------- RecordVideo.m ---------------------
% 脚本：读取 JSON 数据，绘制 UAV 仿真动画并保存为 MP4 视频（非函数）

% ---------- 1. 数据读取 ----------
save_path     = 'C:\Users\pc\Desktop\PythonProject\ConflictResolution\all_train_data\NarrowCorridor_2_train_data';
obstacle_file = fullfile(save_path, 'NarrowCorridor_2_obs.json');
data_file     = fullfile(save_path, 'all_episodes_data_NarrowCorridor_2.json');
target_file   = fullfile(save_path, 'NarrowCorridor_2_target.json');

obstacles   = readJSONFile(obstacle_file);
data        = readJSONFile(data_file);
target_data = readJSONFile(target_file);

% ---------- 2. 参数初始化 ----------
num_uav     = size(target_data, 1);
num_steps   = size(data.s, 1);
sample_time = 0.2;
fixed_area  = 900;
colors = [ ...
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
    0.4660, 0.6740, 0.1880;
    0.3010, 0.7450, 0.9330;
    0.6350, 0.0780, 0.1840;
    0,      0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
];

% ---------- 3. 创建并打开 VideoWriter ----------
videoName = 'uav_simulation.mp4';
v = VideoWriter(videoName, 'MPEG-4');
v.FrameRate = 20;
open(v);

% ---------- 4. 创建图窗并绘制初始静态元素 ----------
figure('Name','UAV Simulation','NumberTitle','off');
hold on; axis equal;
xlabel('X'); ylabel('Y'); title('仿真动画');
% 障碍物：Tag='obstacle'
scatter(obstacles(:,1), obstacles(:,2), 10, 'r', 'filled', 'Tag','obstacle');

% ---------- 5. 循环绘制每帧并写入视频 ----------
for step = 1:num_steps
    % 删除上帧 UAV、椭圆、感知圆
    delete(findobj(gca, 'Tag','uav'));
    delete(findobj(gca, 'Tag','ellipse'));
    delete(findobj(gca, 'Tag','fov'));
    
    % 绘制当前帧所有 UAV
    for uav_idx = 1:num_uav
        ch = 6*(uav_idx-1);
        center  = data.s(step, 1+ch : 2+ch);
        a       = data.s(step, 5+ch);
        b       = fixed_area / a;
        ang_deg = data.s(step, 6+ch) * 180/pi;
        col     = colors(uav_idx,:);
        % 椭圆曲线，Tag='ellipse'
        theta = linspace(0,2*pi,200);
        pts   = ([a*cos(theta); b*sin(theta)]' * ...
                 [cosd(ang_deg), -sind(ang_deg); sind(ang_deg), cosd(ang_deg)])';
        plot(pts(1,:)+center(1), pts(2,:)+center(2), 'LineWidth',2, ...
             'Color',col, 'Tag','ellipse');
        % UAV 位置，Tag='uav'
        scatter(center(1), center(2), 100, '^', 'MarkerEdgeColor',col, ...
                'MarkerFaceColor',col, 'Tag','uav');
        % 感知圆（Field of view），Tag='fov'
        r = 90;
        x_c = center(1) + r*cos(theta);
        y_c = center(2) + r*sin(theta);
        fill(x_c, y_c, col, 'FaceAlpha',0.1, 'EdgeColor',col, ...
             'LineStyle','--', 'Tag','fov');
    end

    title(sprintf('时间: %.1f s', sample_time*step));
    drawnow;
    % 抓帧并写入
    frame = getframe(gcf);
    writeVideo(v, frame);
end

% ---------- 6. 关闭视频对象 ----------
close(v);
fprintf('已保存视频: %s\n', videoName);

% ---------- 本地函数：读取 JSON ----------
function dataOut = readJSONFile(fname)
    if exist(fname,'file')
        txt = fileread(fname);
        dataOut = jsondecode(txt);
    else
        error('无法找到文件: %s', fname);
    end
end
