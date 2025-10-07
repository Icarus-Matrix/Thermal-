%% radiosity network solver 
%% This script will be used to solve the radiosity network for the fuselage and the avionics the contributing componenets are:
    %1. All avionics
    %2. Fuselage
    %3. Atmosphere 
%% This script will be able to take an arbitrary number of surfaces for analysis 

function [J]=getRadiosity(numberSurfs,emissivities,viewFactors,Temperatures,areas)

%%will solve matrix AJ=B
N=numberSurfs; %%number of nodes 

%declare arrays 
A=zeros(N);
B=zeros(N,1);
Rc=zeros(N,1);
Rs=zeros(N);
Eb=zeros(N,1);

%constants
sigma=5.67*10^-8;

%% radisoity buildup 
for i=1:N %loop through each surface
    
    Eb(i)=sigma*Temperatures(i)^4; %black body surface temperature

    %%contact resistance
    if emissivities(i)==1
        %black body handeling J=Eb
        A(i,:)=0;
        A(i,i)=1;
        B(i)=Eb(i);
        continue
    else 
    Rc(i)=(1-emissivities(i))/(emissivities(i)*areas(i));
    %%network setup 
    A(i,i)=A(i,i)+1/Rc(i);
    B(i)=B(i)+Eb(i)/Rc(i);
    end 

    %%loop through each surface for space resistances
    for j=1:N
        if j==i, continue; end
            if viewFactors(i,j)>1e-12
            %space resistances
            Rs(i,j)=1/(areas(i)*viewFactors(i,j));
            A(i,i)=A(i,i)+1/Rs(i,j);
            A(i,j)=A(i,j)-1/Rs(i,j);
            end 
    end
 
end

if rcond(A) < 1e-12
    warning('Matrix A is ill-conditioned. Check view factors or emissivities.')
end

J=linsolve(A,B); %solve for radiosity for each surface

end %end function


        