function cell_new = update_life_universe (x_old)
% update universe in game of life


[maxx, maxy] = size(x_old);

for nx = 1:maxx
    for ny = 1:maxy
        cell_new(nx, ny) = update_life_cell(x_old, nx, ny);
    end
end

end
