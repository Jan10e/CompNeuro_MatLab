function cell_new = update_life_cell(xc, nx, ny)
% update_life_cell will update cell_new(nx, ny) in game of life
% by Daeyeol Lee
% Jan 26, 2017
% xc = current universe

[maxx maxy]=size(xc);

self_alive=xc(nx, ny);
n_alive=0;

for dx=-1:1,
    for dy=-1:1,
        if dx==0 && dy==0, continue;
        end;
        xpos=nx+dx;
        ypos=ny+dy;
        if xpos<=0 || xpos> maxx, continue; end;
        if ypos<=0 || ypos> maxy, continue; end;
        if xc(xpos, ypos)>0, n_alive=n_alive+1; end;
    end;
end;

cell_new=0;
if n_alive==3, cell_new=1;
elseif n_alive==2 && self_alive>0, cell_new=1; 
end;

end
