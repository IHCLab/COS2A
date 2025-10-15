function plot_img_with_sample(random_indices, img_size)

[saved_y, saved_x] = ind2sub([img_size(1), img_size(2)], random_indices);

text_handles = gobjects(length(random_indices), 1);
for i = 1:length(random_indices)
    scatter(saved_x(i), saved_y(i), 100, [0 0.4470 0.7410], 'o', 'filled');
    
    [text_x, text_y] = calcLabelPos(saved_x(i), saved_y(i), img_size(2), img_size(1));
    text_handles(i) = text(text_x, text_y, sprintf('(%d,%d)', saved_x(i), saved_y(i)), ...
        'Color', 'k', 'FontSize', 14, 'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'BackgroundColor', [1 1 1 0.6], 'ButtonDownFcn', @startDragText);
end

set(gcf, 'WindowButtonMotionFcn', @dragText);
set(gcf, 'WindowButtonUpFcn', @stopDragText);
setappdata(gcf, 'dragging', false);
setappdata(gcf, 'current_text', []);
end

%%
function startDragText(src, ~)
fig = ancestor(src, 'figure');
setappdata(fig, 'dragging', true);
setappdata(fig, 'current_text', src);
set(fig, 'Pointer', 'hand');
end

function dragText(~, ~)
fig = gcf;
dragging = getappdata(fig, 'dragging');
current_text = getappdata(fig, 'current_text');

if isequal(dragging, true) && ~isempty(current_text) && ishandle(current_text)
    ax = ancestor(current_text, 'axes');
    cp = get(ax, 'CurrentPoint');
    [img_height, img_width, ~] = size(getimage(ax));
    new_x = max(1, min(img_width, cp(1,1)));
    new_y = max(1, min(img_height, cp(1,2)));
    
    set(current_text, 'Position', [new_x, new_y]);
end
end

function stopDragText(~, ~)
fig = gcf;
if getappdata(fig, 'dragging')
    setappdata(fig, 'dragging', false);
    setappdata(fig, 'current_text', []);
    set(fig, 'Pointer', 'arrow');
end
end

function [text_x, text_y] = calcLabelPos(x, y, img_w, img_h)
offset = 8;
text_w = 60;
text_h = 20;
text_x = x + offset;
text_y = y;

if text_x + text_w > img_w, text_x = x - text_w - offset; end
if text_y + text_h > img_h, text_y = y - text_h - offset; end
if text_y < 1, text_y = y + offset; end
if text_x < 1, text_x = x + offset; end
end
