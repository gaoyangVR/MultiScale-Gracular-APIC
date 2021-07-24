#version 330

uniform vec3 eyepos;		//world coordinate.
in vec3 worldpos;
uniform vec3 lightpos;// = vec3(2.0f,0,0);
uniform sampler3D densetex;

struct Ray {
	vec3 origin;
	vec3 dir;
};

struct AABB {
	vec3 Min;
	vec3 Max;
};

const int maxstep = 300;
const int maxlstep = 300;
const float maxdis = sqrt(3.0f);
const float tstep = maxdis/maxstep;
const float lstep = maxdis/maxlstep;
const vec3 lightIntensity = vec3(50.0f);
const float Absorption = 1.0f;


bool IntersectBox(Ray r, AABB aabb, out float tnear, out float tfar)
{
	vec3 invR = 1.0 / r.dir;
	vec3 tbot = invR * (aabb.Min-r.origin);
	vec3 ttop = invR * (aabb.Max-r.origin);
	vec3 tmin = min(ttop, tbot);
	vec3 tmax = max(ttop, tbot);
	vec2 t = max(tmin.xx, tmin.yz);
	tnear = max(t.x, t.y);
	t = min(tmax.xx, tmax.yz);
	tfar = min(t.x, t.y);
	return tnear <= tfar;
}

void main()
{
	Ray ray;
	ray.dir = normalize(worldpos-eyepos);
	ray.origin = eyepos;
	float tnear, tfar;
	AABB aabb = AABB(vec3(0.0f), vec3(1.0f) );
	IntersectBox(ray, aabb, tnear, tfar);
	if( tnear<0.0f )
		tnear = 0.0f;

	vec3 pos = ray.origin + tnear*ray.dir;
	vec3 step = ray.dir*tstep;
	float dis = tnear;
	float T = 1.0f;		//tran
	vec3 lightdir;
	vec3 lsample;
	vec3 Lo = vec3(0.0f);
	for( int i=1; i<maxstep && dis<tfar; i++, pos += step, dis += tstep )
	{
		//pos's density
		float density = texture(densetex, pos).x;
		if( density<=0 ) continue;
 		T *= 1-density*tstep*Absorption;
 		if( T<0.01 ) break;
		//light
		float Tl = 1.0f;
// 		{			//这一段是光照的，比较耗时，平时可以关闭
// 			lightdir = normalize( lightpos-pos )*lstep;
// 			lsample = pos + lightdir;
// 			for(int j=1; j<maxlstep; j++ )
// 			{
// 				float ldens = texture( densetex, lsample);
// 				Tl *= 1-ldens*lstep*Absorption;		//absorption决定光穿过的阻力
// 				if( Tl<0.01 ) break;
// 				lsample += lightdir;
// 				if( lsample.x<0 || lsample.y<0 || lsample.z < 0 || lsample.x>1 || lsample.y>1 || lsample.z>1 )
// 					break;
// 			}
// 		}

		vec3 Li = lightIntensity * Tl;	//lightIntensity决定光的强度
		Lo += Li*T*density*tstep;
	}

	//gl_FragColor = vec4( T, 0, 0, 1 );
	gl_FragColor = vec4( Lo, 1-T );
}
