%% montecarlo simulation 
%assumptions 
%1.The fuselage is treated as a blackbox. All rays that miss other
%componenets are absorbed by the fuselage. 
%2. Avionics are assumed to be rectangular surfaces
%3. Rays are launch with random position and direciton 

function viewFactorMatrix = avionicsViewFactors(Nrays)
    % Cylinder fuselage dimensions
    R_fuse = 175;    % mm, radius
    L_fuse = 1154;    % mm, length (x direction)
    
    %read file 
    avionics=readtable("C:\Users\BrianByrne\Desktop\Thermal Analysis\VIEWFACTOR GEOMETRY.xlsx","Sheet","Avionics Geometry");

    % Avionics data (xc,yc,zc,w,h)
    % Taken from table (zc=0 for all, assumed placed at tray midplane)
    Ncomp = size(avionics,1);
    areas=avionics{:,10}; %m^2
    sum=0;

    %initalize
    viewFactorMatrix = zeros(Ncomp); % last row/col = fuselage
 
    for i=1:Ncomp-1 %%loops through each componenet and finds view factor 
        
        %%determine what faces to skip based on what view factor is being
        %%calcualted 
        %%determine what faces to skip based on what view factor is being calculated
        if i <= 6 % Payload
            skipMe = [1 2 3 4 5 6];
        elseif i >= 7 && i <= 12 % Power Brick
            skipMe = [7 8 9 10 11 12];
        elseif i >= 13 && i <= 18 % Auterion
            skipMe = [13 14 15 16 17 18];
        elseif i == 19 % GPS
            skipMe = 19;
        elseif i == 20 % Ethernet
            skipMe = 20;
        elseif i >= 21 && i <= 26 % Radio
            skipMe = [21 22 23 24 25 26];
        elseif i >= 27 && i <= 32 % Battery 1
            skipMe = [27 28 29 30 31 32];
        elseif i >= 33 && i <= 38 % Battery 2
            skipMe = [33 34 35 36 37 38];
        elseif i >= 39 && i <= 44 % Battery 3
            skipMe = [39 40 41 42 43 44];
        elseif i >= 45 && i <= 50 % Battery 4
            skipMe = [45 46 47 48 49 50];
        elseif i>=51 && i<=56 
            skipMe = [51 52 53 54 55 56]; %silvus
        elseif i==57 %avionic tray 
            skipMe = 57;
        end


        hits=zeros(Ncomp,1); %initalize hits for each run 

        %% grab geometry information
        xc = (avionics{i,2});yc = (avionics{i,3});zc = (avionics{i,4});   
        w  = (avionics{i,5});h  = (avionics{i,6});   
        nx = (avionics{i,7});ny = (avionics{i,8});nz = (avionics{i,9});   

         for n = 1:Nrays
            % Sample random point on rectangle surface
            [x,y,z]=getRandomPoints(xc,yc,zc,w,h,nx,ny,nz);

            % Lambertian random direction (normal = +z)
            [dx,dy,dz] = lambertianDirection(nx,ny,nz);
            
            
            % First check intersection with fuselage
            [hitFuse,tfuse] = rayCylinderIntersect(x,y,z,dx,dy,dz,R_fuse,L_fuse);

            % Track nearest hit (fuselage or another component)
            nearestT = inf;
            hitIndex = [];

            if hitFuse
                nearestT = tfuse;
                hitIndex = Ncomp; % fuselage index
            end

            %% Check intersections with other components
            for j = 1:Ncomp
                
                %%skip Fii 
                    if ismember(j,skipMe)
                        continue;
                    end 

                %grab geometry for componenet
                xcj = (avionics{j,2});   ycj = (avionics{j,3});   zcj = (avionics{j,4});   
                wj  = (avionics{j,5});   hj  = (avionics{j,6});   
                nxj = (avionics{j,7});   nyj = (avionics{j,8});   nzj = (avionics{j,9});  
                
                %determine if ray hits comoponent
                [hitRect, tj] = rayRectangleIntersect(x,y,z,dx,dy,dz,xcj,ycj,zcj,wj,hj,nxj,nyj,nzj);
                if hitRect && tj < nearestT
                    nearestT = tj;
                    hitIndex = j;
                end
            end

            %Hit counter 
            if ~isempty(hitIndex)
                hits(hitIndex) = hits(hitIndex) + 1;
            end
         end %end avionic vf loop 

         %compute view factors for current avionic componenet
        viewFactorMatrix(i,:)=hits(:)/Nrays;

    end %end componenet loop 
    
    %%apply reciprocity to find fuselage view factors 
    fuselageArea=2*pi*R_fuse*L_fuse*0.000001;
    for i=1:Ncomp
        if i==Ncomp
           viewFactorMatrix(Ncomp,i)=1-sum;
        else 
        viewFactorMatrix(Ncomp,i)=viewFactorMatrix(i,Ncomp)*(areas(i)/fuselageArea);
        end 
        sum=sum+viewFactorMatrix(Ncomp,i);
    end 
    % Display matrix
    disp('View Factor Matrix:')
    disp(viewFactorMatrix)
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

function [hit,tmin] = rayCylinderIntersect(x0,y0,z0,dx,dy,dz,R,L)
    hit  = false;
    tmin = inf;

    % --- Curved wall check ---
    A = dy^2 + dz^2;
    B = 2*(y0*dy + z0*dz);
    C = y0^2 + z0^2 - R^2;

    if A > 1e-12
        t = roots([A B C]);
        t = t(t>0); % forward only
        if ~isempty(t)
            for k = 1:length(t)
                xi = x0 + dx*t(k);
                if abs(xi) <= L/2 && t(k) < tmin
                    hit = true;
                    tmin = t(k);
                end
            end
        end
    end

    % --- Endcap checks ---
    if abs(dx) > 1e-12
        % Left cap (x=-L/2)
        tcap = (-L/2 - x0)/dx;
        if tcap > 0
            ycap = y0 + dy*tcap;
            zcap = z0 + dz*tcap;
            if (ycap^2 + zcap^2) <= R^2 && tcap < tmin
                hit = true;
                tmin = tcap;
            end
        end

        % Right cap (x=+L/2)
        tcap = (L/2 - x0)/dx;
        if tcap > 0
            ycap = y0 + dy*tcap;
            zcap = z0 + dz*tcap;
            if (ycap^2 + zcap^2) <= R^2 && tcap < tmin
                hit = true;
                tmin = tcap;
            end
        end
    end
end

%% avionic intersection
function [hit,t] = rayRectangleIntersect(x0,y0,z0,dx,dy,dz,xc,yc,zc,w,h,nx,ny,nz)
    
    hit=false;
    t=inf;

    if nz==1 || nz==-1 %z normal surfaces
        t=(zc-z0)/dz;
        xi=x0+t*dx; yi=y0+t*dy;
        if abs(xi - xc) <= w/2 && abs(yi - yc) <= h/2
                hit = true;
        end

    elseif ny==1 || ny==-1 %%y normal surfaces
          t = (yc - y0)/dy;
          xi = x0 + dx*t; zi = z0 + dz*t;
          if abs(xi - xc) <= w/2 && abs(zi - zc) <= h/2
                hit = true;
          end

    elseif nx==1 || nx==-1 %x normal surfaces
        t = (xc - x0)/dx;
        yi = y0 + dy*t; zi = z0 + dz*t;
        if abs(yi - yc) <= w/2 && abs(zi - zc) <= h/2
                hit = true;
        end
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

