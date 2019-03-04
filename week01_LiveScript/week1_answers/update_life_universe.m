function x_new = update_life_universe( x_old )
% updates the universe in game of life

[maxx maxy]=size(x_old);

for nx=1:maxx,
    for ny=1:maxy,
        x_new(nx, ny)=update_life_cell(x_old, nx, ny);
    end;
end;

end

