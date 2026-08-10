function k=getPropConvKer(lambda,z_stp,x_max,x_stp ,is_cyclic)

if~exist('is_cyclic','var')
    is_cyclic=0;
end
if ~is_cyclic
    bdr_fact=2;
else
    bdr_fact=1;
end
wx_max=x_max*bdr_fact;
x_grid=[-wx_max:x_stp:wx_max];
[x_grid,y_grid]=ndgrid(x_grid);

v_max=lambda/(2*x_stp);


v_stp=lambda/(2*wx_max);
v_grid=[-v_max:v_stp:v_max];
[vx,vy]=ndgrid(v_grid);
r=(vx.^2+vy.^2);
mask=r<0.5^2;  %Can adjust this threshold 
p=sqrt(1-r.*mask);
k=exp(-2*pi*i/lambda*(p.*mask)*z_stp).*mask;

return
