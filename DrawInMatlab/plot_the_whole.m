
    % MATLAB 绘图脚本：根据集合中 UAV 成员来确定颜色（动画版）
    % 自定义函数：读取 JSON 文件
    
    % ---------------------  1. 数据读取  ---------------------
    % 设置数据文件路径
    save_path = 'C:\Users\pc\Desktop\PythonProject\ConflictResolution\all_train_data\NarrowCorridor_2_train_data';
    obstacle_file = fullfile(save_path, 'NarrowCorridor_2_obs.json');
    data_file = fullfile(save_path, 'all_episodes_data_NarrowCorridor_2.json');
    target_file = fullfile(save_path, 'NarrowCorridor_2_target.json');
    % 读取障碍物数据
    obstacles = readJSONFile(obstacle_file);
    % 读取仿真数据
    data = readJSONFile(data_file);
    % 读取目标数据
    target_data = readJSONFile(target_file);

    % --------------------- 2. 初始化参数 ---------------------
    % 确定 UAV 数量
    num_uav = size(target_data, 1);
    
    % 确定时间步数量
    num_steps = size(data.s, 1);
    sample_time = 0.2;
    % 分配颜色给每个 UAV
    % 使用 MATLAB 的预定义颜色映射（如 lines）
    color = [
    0.8500, 0.3250, 0.0980;  % 橙色
    0.9290, 0.6940, 0.1250;  % 黄色
    0.4940, 0.1840, 0.5560;  % 紫色
    0.4660, 0.6740, 0.1880;  % 绿色
    0.3010, 0.7450, 0.9330;  % 浅蓝
    0.6350, 0.0780, 0.1840;  % 深红
    0, 0.4470, 0.7410;        % 深蓝
    0.8500, 0.3250, 0.0980;   % 橙色（重复或新增颜色）
    0.9290, 0.6940, 0.1250;   % 黄色
    0.4940, 0.1840, 0.5560;   % 紫色
];
    
    % 计算短半轴长度 b，使得椭圆面积固定为 900*pi
    fixed_area = 900 ;
    
    % ---------------------  4. 开始绘图  ---------------------
    figure('Name','仿真动画','NumberTitle','off');
    hold on;
    axis equal;
    % xlim([-100, 900]);
    % ylim([-200, 200]);
    xlabel('X');
    ylabel('Y');
    title('仿真动画');

    % 先绘制障碍物（静态）
    scatter(obstacles(:,1), obstacles(:,2), 10, 'r', 'filled', 'DisplayName', '障碍物');

    % 循环动画
    for step = 1:num_steps
        % 删除前一帧（除了障碍物）
        if step > 1
            delete(findobj(gca, 'Type', 'line'));
            delete(findobj(gca, 'Type', 'scatter', 'Marker', '^')); % UAV
            delete(findobj(gca, 'Type', 'scatter', 'Marker', 'o')); % Robots
            delete(findobj(gca, 'Type', 'patch')); % 椭圆与填充圆
        end
        axis equal;
        % xlim([-100, 900]);
        % ylim([-200, 200]);
        xlabel('X');
        ylabel('Y');
        title('仿真动画');

        % 先绘制障碍物（静态）
        scatter(obstacles(:,1), obstacles(:,2), 10, 'r', 'filled', 'DisplayName', '障碍物');
        current_time = sample_time * step;
        % 绘制每个集合
        for uav_idx = 1:num_uav
            channel = 6*(uav_idx-1);
            % 1) 绘制虚拟区域椭圆
            center = data.s(step,1+channel:2+channel);
            a = data.s(step,5+channel);
            b = fixed_area / a;
            angle_deg = data.s(step,6+channel) * 180 / pi;  
            plot_ellipse(center, a, b, angle_deg, color(uav_idx,:));
            % 绘制 UAV
            scatter(data.s(step,1+channel), data.s(step,2+channel), 100, '^', ...
                'MarkerEdgeColor', color(uav_idx,:), 'MarkerFaceColor', color(uav_idx,:));

            % 绘制感知区域(填充圆)
            r = 90;  
            theta = linspace(0, 2*pi, 100);
            x_circle = data.s(step,1+channel) + r * cos(theta);
            y_circle = data.s(step,2+channel) + r * sin(theta);
            fill(x_circle, y_circle, color(uav_idx,:), ...
                 'FaceAlpha', 0.1, 'EdgeColor', color(uav_idx,:), 'LineStyle','--','LineWidth',1);

            % % 绘制分配给 UAV 的机器人
            % assigned_robots = data.robots(step,:);
            % for robot_i = 1:length(assigned_robots)
            %     robot = assigned_robots(robot_i);
            %     if ismember(set_info.set_id, robot.assigned_uav)
            %         scatter(robot.position(1), robot.position(2), 50, 'o', ...
            %             'MarkerEdgeColor', color, 'MarkerFaceColor', color, ...
            %             'DisplayName', sprintf('机器人 %d', robot.id));
                 
        end

        title(['时间: ', num2str(current_time), 's']);
        drawnow;
        pause(0.05);
    end

    hold off;




%% --------------------- 辅助函数：绘制椭圆 ---------------------
function h = plot_ellipse(center, a, b, angle_deg, color)
    theta = linspace(0, 2*pi, 100);
    x = a * cos(theta);
    y = b * sin(theta);
    angle_rad = deg2rad(angle_deg);
    R = [cos(angle_rad), -sin(angle_rad); sin(angle_rad), cos(angle_rad)];
    ellipse_coords = R * [x; y];
    x_ellipse = ellipse_coords(1,:) + center(1);
    y_ellipse = ellipse_coords(2,:) + center(2);
    h = plot(x_ellipse, y_ellipse, 'Color', color, 'LineWidth', 2);
end
%% --------------------- 辅助函数：读取json ---------------------
function jsonData = readJSONFile(fileName)
        if exist(fileName, 'file')
            fid = fopen(fileName, 'r');
            raw = fread(fid, inf); % 读取所有文件内容
            str = char(raw'); % 转换为字符数组
            fclose(fid); % 关闭文件
            jsonData = jsondecode(str); % 解码 JSON 数据
        else
            error('文件不存在：%s', fileName);
        end
    end
