function [random_indices] = plot_sample_curve(scene,random_mode, X, Y, Z_fused_COS2A, Z_fused_Universal)

% load
Y_H_2D = reshape(X, [], size(X, 3))';
Y_S_2D = reshape(Y, [], size(Y, 3))'; 
Z_fused_COS2A_2D = reshape(Z_fused_COS2A, [], size(Z_fused_COS2A, 3))'; 
Z_fused_Universal_2D = reshape(Z_fused_Universal, [], size(Z_fused_Universal, 3))'; 

% random pixel
if random_mode == 1
    num_random_pixels = 3;
    random_indices = randperm(size(Y_H_2D, 2), num_random_pixels);
    Y_S_sample = Y_S_2D(:, random_indices);
    Y_H_sample = Y_H_2D(:, random_indices);
    Z_fused_COS2A_sample = Z_fused_COS2A_2D(:, random_indices);
    Z_fused_Universal_sample = Z_fused_Universal_2D(:, random_indices);

elseif random_mode == 2
    xy_all = {
        [252,125; 219,23; 31,249; 15,113; 79,39];       % 1: farm
    };
    if scene <= numel(xy_all)
        xy_list = xy_all{scene};
    else
        xy_list = xy_all{1};
    end
    random_indices = sub2ind([size(X,1), size(X,2)], xy_list(:, 2), xy_list(:, 1));
    num_random_pixels = size(xy_list, 1);

    % GT Scaling scaler
    S2_band_idx_in_AVI = [1,4,11,23,28,32,36,41,45,53,110,153];
    Y_S_sample = Y_S_2D(:, random_indices);
    Y_H_sample = Y_H_2D(:, random_indices);
    Y_H_sample_12 = Y_H_sample(S2_band_idx_in_AVI, :);        
    Z_fused_COS2A_sample = Z_fused_COS2A_2D(:, random_indices);
    Z_fused_Universal_sample = Z_fused_Universal_2D(:, random_indices);
    aalpha_list = zeros(1, num_random_pixels);
    for i = 1:length(random_indices)
        gt = Y_H_sample_12(:, i);
        y_s = Y_S_sample(:, i); 
        aalpha_list(1, i) = lsqnonneg(gt, y_s);
    end
end

% band
AVI_wavelength = [ 365.92981000000003 , 375.59398999999996 , 385.26254 , 394.93552 , 404.61288 , 414.29462 , 423.98077 , 433.6713 , 443.36620999999997 , 453.06552 , 462.7692 , 472.47729 , 482.18976000000004 , 491.90665 , 501.6279 , 511.35355000000004 , 521.08356 , 530.81799 , 540.55682 , 550.30005 , 560.04767 , 569.79962 , 579.55603 , 589.31677 , 599.08191 , 608.8515 , 618.62543 , 628.4037500000001 , 638.18646 , 647.97357 , 654.7923 , 657.76508 , 664.59937 , 667.56097 ,   674.40125 , 684.19794 , 693.98938 , 703.77563 , 713.55664 , 723.33252 , 733.10309 , 742.86853 , 752.62872 , 762.38373 , 772.1334800000001 , 781.87805 , 791.6174299999999 , 801.35156 , 811.08051 , 820.80426 , 830.5227699999999 , 840.23608 , 849.94415 , 859.6470899999999 , 869.34479 , 879.03723 , 888.72449 , 898.40656 , 908.08337 , 917.75507 , 927.4214499999999 , 937.0827 , 946.73871 , 956.38947 , 966.0351 , 975.67548 , 985.31061 , 994.94055 , 1004.5653000000001 , 1014.1848799999999 , 1023.7991900000001 , 1033.4083300000002 , 1043.01221 , 1052.61096 , 1062.20447 , 1071.79272 , 1081.37573 , 1090.9536099999998 , 1100.52625 , 1110.09375 , 1119.65601 , 1129.21301 , 1138.76477 , 1148.3114 , 1157.85278 , 1167.38904 , 1176.91992 , 1186.4458 , 1195.96631 , 1205.48169 , 1214.99182 , 1224.4967 , 1233.99646 , 1243.49097 , 1252.77295 , 1252.98022 , 1262.4643600000002 , 1262.74573 , 1272.71826 , 1282.69055 , 1292.66248 , 1302.63428 , 1312.6058300000002 , 1322.57715 , 1332.5481 , 1342.51892 , 1352.4895 , 1362.45984 , 1372.42993 , 1382.3997800000002 , 1392.36938 , 1402.33875 , 1412.3078600000001 , 1422.27673 , 1432.2453600000001 , 1442.2137500000001 , 1452.1818799999999 , 1462.1499 , 1472.1175500000002 , 1482.08496 , 1492.05212 , 1502.01904 , 1511.9858399999998 , 1521.95227 , 1531.9184599999999 , 1541.88452 , 1551.85022 , 1561.8156700000002 , 1571.78101 , 1581.74597 , 1591.71082 , 1601.67529 , 1611.63965 , 1621.60364 , 1631.5675 , 1641.53101 , 1651.4943799999999 , 1661.45752 , 1671.42029 , 1681.38293 , 1691.34534 , 1701.30737 , 1711.26929 , 1721.2309599999999 , 1731.19238 , 1741.15344 , 1751.11438 , 1761.0750699999999 , 1771.0355200000001 , 1780.99573 , 1790.95569 , 1800.91541 , 1810.87488 , 1820.83411 , 1830.7930900000001 , 1840.7518300000002 , 1850.71033 , 1860.66858 , 1870.62659 , 1871.7843 , 1865.96375 , 1876.0252699999999 , 1886.0845900000002 , 1896.14148 , 1906.19604 , 1916.24841 , 1926.29846 , 1936.3460699999998 , 1946.39148 , 1956.4345700000001 , 1966.47534 , 1976.51379 , 1986.55005 , 1996.5838600000002 , 2006.61536 , 2016.64465 , 2026.67163 , 2036.69617 , 2046.71863 , 2056.73877 , 2066.75635 , 2076.77173 , 2086.78491 , 2096.7956499999996 , 2106.8042 , 2116.8103 , 2126.81421 , 2136.81567 , 2146.81494 , 2156.81201 , 2166.80664 , 2176.79907 , 2186.78906 , 2196.7766100000003 , 2206.76221 , 2216.74536 , 2226.7260699999997 , 2236.70459 , 2246.68066 , 2256.65454 , 2266.62622 , 2276.59546 , 2286.5625 , 2296.5271000000002 , 2306.4895 , 2316.44946 , 2326.4072300000003 , 2336.36255 , 2346.31567 , 2356.2666 , 2366.2150899999997 , 2376.16113 , 2386.10522 , 2396.04663 , 2405.98608 , 2415.9231 , 2425.85767 , 2435.7900400000003 , 2445.7199699999996 , 2455.64771 , 2465.5732399999997 , 2475.4963399999997 , 2485.4172399999998 , 2495.33569];
removed_idx = [1:10, 104:116, 152:170, 215:224]; valid_idx = setdiff(1:224, removed_idx);
AVI_wavelength = AVI_wavelength(valid_idx) / 1000; 
S2_band_idx_in_AVI = [1,4,11,23,28,32,36,41,45,53,110,153];
S2_wavelength = AVI_wavelength(S2_band_idx_in_AVI);
seg1 = AVI_wavelength(AVI_wavelength < 1.4);
seg2 = AVI_wavelength(AVI_wavelength > 1.4 & AVI_wavelength < 1.8);
seg3 = AVI_wavelength(AVI_wavelength > 1.8);
x_axis = [seg1, seg2, seg3];

% plot
for i = 1:num_random_pixels
    [y, x] = ind2sub([size(X,1), size(X,2)], random_indices(i));

    figure('Position', [100, 100, 800, 300]);
    Y_H_i = aalpha_list(1, i)*Y_H_sample(:, i); % GT Scaling
    Y_S_i = Y_S_sample(:, i);
    Z_fused_COS2A_i = Z_fused_COS2A_sample(:, i);
    Z_fused_Universal_i = Z_fused_Universal_sample(:, i);

    plot(x_axis, Y_H_i, 'k-', 'LineWidth', 1.5); hold on;    
    plot(x_axis, Z_fused_COS2A_i, 'r-', 'LineWidth', 1.5)
    plot(x_axis, Z_fused_Universal_i, 'Color', [0, 0.5, 0], 'LineWidth', 1.5);
    
    for ii = 1:length(S2_wavelength)
        plot([S2_wavelength(ii), S2_wavelength(ii)], [0, Y_S_i(ii)], 'b-', 'LineWidth', 1.5); % Sentinel-2
    end

    font_size = 17; font_name = 'Times New Roman0';
    tick_positions = [seg1(1),1.0, 1.5,2, seg3(end)]; xticks(tick_positions);
    xticklabels({'0.5','1','1.5','2','  2.5'});
    xlim([min(x_axis)-0.05, max(x_axis)+0.05]);
    lgd = legend('AVIRIS', 'COS2A', 'Universal Model','Sentinel-2', 'Location', 'northeast'); 
    defaultTokenSize = lgd.ItemTokenSize;
    set(lgd, 'FontSize',10, 'FontName', font_name, 'ItemTokenSize', [10, defaultTokenSize(2)]);
    ylabel('Reflectance', 'FontSize', font_size, 'FontName', font_name);
    xlabel('Wavelength (m)', 'FontSize', font_size, 'FontName', font_name);
    title(sprintf('Pixel Value at (%d,%d)', x, y), 'FontSize', font_size, 'FontName', font_name);

    ax = gca;
    ax.FontSize = font_size;
    ax.FontName = font_name; 
    grid on; 
end
