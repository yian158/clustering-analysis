function [  ] = plot_2d_Data( dataSet, cluster_label )

        Style = {'+', 'o', 'x', 'd', '*', 'v', 'p' };
        Colors1 = get(gca,'colororder');
        
        hold on
        for j = 1:numel(unique(cluster_label))            
            plot(dataSet(cluster_label==j,1), dataSet(cluster_label==j,2), Style{j}, 'MarkerSize',5,  'Color', Colors1(j, :), 'linewidth', 1.5);
            set(gca, 'FontSize',10);
            set(gca, 'LineWidth',1.5);
            set(gca,'xcolor','k');
            set(gca,'ycolor','k');
            box on
        end
        min_x = min(dataSet(:,1));
        max_x = max(dataSet(:,1));
        p_x = (max_x-min_x)*0.1;
        min_y = min(dataSet(:,2));
        max_y = max(dataSet(:,2));
        p_y = (max_y-min_y)*0.1;       
        
        axis([min_x-p_x  max_x+p_x  min_y-p_y  max_y+p_y]);
end

