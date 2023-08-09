function points = lineSphereIntersection(x0,y0,z0,dx,dy,dz,cx,cy,cz,R)

l = [dx dy dz];
c = [cx cy cz];
r = R;
o = [x0 y0 z0];

d_inter = sum(l.*(o-c),2).^2-(sum((o-c).^2,2)-r.^2);

d_inter(d_inter<=0) = nan;



d = -(sum(l.*(o-c),2)) + sqrt(d_inter);


points = o + d.*l;












end