close all;
Nrays=10;

viewfactorMatrix=avionicsViewFactors(Nrays);

Nrays=10;
%% debug visualization 
avionics = readtable("VIEWFACTOR GEOMETRY.xlsx");

    for i = 1:57
        xc = (avionics{i,2});yc = (avionics{i,3});zc = (avionics{i,4});   
        w  = (avionics{i,5});h  = (avionics{i,6});   
        nx = (avionics{i,7});ny = (avionics{i,8});nz = (avionics{i,9});   

        % Build surface grid depending on orientation
        if abs(nz) == 1 % XY plane
            [X,Y] = meshgrid(linspace(xc-w/2,xc+w/2,2), linspace(yc-h/2,yc+h/2,2));
            Z = zc*ones(size(X));
        elseif abs(ny) == 1 % XZ plane
            [X,Z] = meshgrid(linspace(xc-w/2,xc+w/2,2), linspace(zc-h/2,zc+h/2,2));
            Y = yc*ones(size(X));
        elseif abs(nx) == 1 % YZ plane
            [Y,Z] = meshgrid(linspace(yc-w/2,yc+w/2,2), linspace(zc-h/2,zc+h/2,2));
            X = xc*ones(size(Y));
        else
            continue
        end

        % Plot surface

        surf(X,Y,Z)
        hold on 
        % Sample a few rays and plot them with quiver3
        for n = 1:Nrays
            [x,y,z] = getRandomPoints(xc,yc,zc,w,h,nx,ny,nz);
            [dx,dy,dz] = lambertianDirection(nx,ny,nz); % your improved version
            quiver3(x,y,z,dx,dy,dz,20,'r') % scale factor=0.05
        end

    end

%function will generator random points for a given surface during iteration
function [x,y,z]=getRandomPoints(xc,yc,zc,w,h,nx,ny,nz)    
    if nz==1 || nz==-1 %z normal surfaces
         x = xc + (rand-0.5)*w;
         y = yc + (rand-0.5)*h;
         z = zc;
    elseif ny==1 || ny==-1%y normal surfaces
         x = xc + (rand-0.5)*w;
         y = yc;
         z = zc + (rand-0.5)*h;
    elseif nx==1 || nx==-1 %x normal surfaces 
         x = xc;
         y = yc + (rand-0.5)*w;
         z = zc + (rand-0.5)*h;
    end 
end 

function [dx,dy,dz] = lambertianDirection(nx,ny,nz)
    %RNG angles
    r1 = rand; r2 = rand;
    theta = 2*pi*r1;
    phi   = asin(sqrt(r2)); % cosine-weighted

    if nz==1 || nz==-1 %z normal surfaces 
        dx = cos(theta)*sin(phi);
        dy = sin(theta)*sin(phi);
        dz = cos(phi)*nz;
    elseif ny==1 || ny==-1 %y normal surfaces 
        dx = cos(theta)*sin(phi);
        dy = cos(phi)*ny;
        dz = sin(theta)*sin(phi);
    elseif nx==1 || nx==-1%x normal surfaces 
        dx=cos(phi)*nx;
        dy= sin(theta)*sin(phi);
        dz = cos(theta)*sin(phi);
    end 
end







