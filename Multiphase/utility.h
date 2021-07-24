#ifndef UTILITY_H
#define UTILITY_H
#include<vector_types.h>

// #define NX 24
// #define NY 24
// #define NZ 96
#define MAXITER 200
#define FLIP_ALPHA 0.8f//0.95f
#define M_PI       3.14159265358979323846
const float DEGtoRAD = 3.1415926f / 180.0f;

#define TYPEFLUID 0
#define TYPEAIR 1
#define TYPEBOUNDARY 2
#define TYPEVACUUM 3
#define TYPEAIRSOLO 4
#define TYPESOLID 5
#define TYPECNT 6
#define TYPESMOKE 7
#define TYPERIGID 8
#define TYPEYIELD 9
#define EMPTYCELL 10
#define TYPESPRAY 11

typedef unsigned int uint;
#define CELL_UNDEF 0xffffffff
#define  NTHREADS 32
#define UNDEF_TEMPERATURE -10000.0f

struct FlipConstant{
	int gnnum;
	int gnum;
	int3 gvnum;
	float samplespace;
	float dt;
	float3 gravity;
	float3 gmin, gmax, cellsize;
	float m0;
	float airm0;
	float waterrho, solidrho, sandrho;
	float sin_phi;//friction angle within sand
	float miu;//boundary friction coefficient
	float cohesion;
	float centripetal, tangential;

	float pradius;
	float3	triHashSize, triHashRes;		//triHashSize是HASH网格的大小;  triHashRes是每一个维度上有几个HASH网格,程序执行过程中不再变化
	float3 t_min, t_max;
	int triHashCells;			//预留的hash数组大小，程序执行过程中不再变化
	//for SPH-like part
	float poly6kern, spikykern, lapkern;

	//marching cube
	//int gridresMC;
};

struct matrix3x3 // __inline__ __host__ __device__ deleted
{
	float x00, x01, x02;
	float x10, x11, x12;
	float x20, x21, x22;
};
struct matrix5x5
{
	float x00, x01, x02, x03, x04;
	float x10, x11, x12, x13, x14;
	float x20, x21, x22, x23, x24;
};

struct matarray{
	matrix3x3* data;//strain rate tensor
	int xn, yn, zn;
	matarray();
	void setdim(int _xn, int _yn, int _zn){ xn = _xn, yn = _yn, zn = _zn; }

	__host__ __device__ inline matrix3x3 &operator ()(int i, int j, int k)
	{
		return data[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline matrix3x3 &operator ()(int i)
	{
		return data[i];
	}
	__host__ __device__ inline matrix3x3 &operator [](int i)
	{
		return data[i];
	}
};
struct f3array{
	float3* data;//v in
	//for APIC
	float* gmass;//m in
	int xn, yn, zn;
	f3array();
	void setdim(int _xn, int _yn, int _zn){ xn = _xn, yn = _yn, zn = _zn; }

	
	__host__ __device__ inline float &operator ()(int i, int j, int k, int type, int mark)
	{
		if (type == 2)
			return gmass[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline float3 &operator ()(int i, int j, int k)
	{
		return data[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline float3 &operator ()(int i)
	{
		return data[i];
	}
	__host__ __device__ inline float3 &operator [](int i)
	{
		return data[i];
	}
};

//创建一个方便转换1维与3维数组的数据结构
struct farray{
	float* data;//v in
	//for APIC
	float* gmass;//m in
	float3* gpos;//x in
	int xn, yn, zn;
	farray();
	void setdim(int _xn, int _yn, int _zn){ xn = _xn, yn = _yn, zn = _zn; }

	__host__ __device__ inline float3 &operator ()(int i, int j, int k, int type)
	{
		if (type == 1)
			return gpos[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline float &operator ()(int i, int j, int k, int type, int mark)
	{
		if (type == 2)
			return gmass[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline float &operator ()(int i, int j, int k)
	{
		return data[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline float &operator ()(int i)
	{
		return data[i];
	}
	__host__ __device__ inline float &operator [](int i)
	{
		return data[i];
	}
};

//创建一个方便转换1维与3维数组的数据结构
struct charray{
	char* data;
	int xn, yn, zn;
	charray();//{ data = NULL; /*xn=NX; yn=NY; zn=NZ;*/}
	void setdim(int _xn, int _yn, int _zn){ xn = _xn, yn = _yn, zn = _zn; }

	__host__ __device__ inline char &operator ()(int i, int j, int k)
	{
		return data[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline char &operator ()(int i)
	{
		return data[i];
	}
	__host__ __device__ inline char &operator [](int i)
	{
		return data[i];
	}
};

__host__ __device__ inline void getijk(int &i, int &j, int &k, int &idx, int w, int h, int d)
{
	i = idx / d / h;
	j = idx / d%h;
	k = idx%d;
}

enum ERENDERMODE{
	RENDER_PARTICLE = 0,
	RENDER_MC,
	RENDER_GRID,
	RENDER_ALL,
	RENDER_CNT
};

enum SIMULATIONMODE{
	SIMULATION_WATER = 0,
	SIMULATION_SOLIDCOUPLING,
	SIMULATION_SMOKE,
	SIMULATION_BUBBLE,
	SIMULATION_HEATONLY,
	SIMULATION_CNT,
	SIMULATION_APICBASIC
};

enum SCENE{
	SCENE_FLUIDSPHERE = 0,
	SCENE_SMOKE,
	SCENE_BOILING,
	SCENE_BOILING_HIGHRES,
	SCENE_MULTIBUBBLE,
	SCENE_DAMBREAK,
	SCENE_MELTING,
	SCENE_MELTINGPOUR,		//melting simulation by pouring water.
	SCENE_FREEZING,
	SCENE_INTERACTION,			//interact with small bubbles, i.e., sub-grid bubbles.
	SCENE_INTERACTION_HIGHRES,			//interact with small bubbles, i.e., sub-grid bubbles.
	SCENE_MELTANDBOIL,		//interact with big bubble
	SCENE_MELTANDBOIL_HIGHRES,		//interact with big bubble
	SCENE_HEATTRANSFER,
	SCENE_CNT,
	SCENE_ALL,
	SCENE_APIC,
	SCENE_LANDSLIDE,
	SCENE_POURING,
	SCENE_SANDSTORM,
	SCENE_EXPLOSION
};

enum VELOCITYMODEL{
	FLIP = 0,
	CIP,
	HYBRID,
	APIC,
	VELOCITYMODEL_CNT
};

enum ECOLORMODE{
	COLOR_PRESS = 0,
	COLOR_UX,
	COLOR_UY,
	COLOR_UZ,
	COLOR_DIV,	//4
	COLOR_PHI,
	COLOR_MARK,	//6
	COLOR_LS,	//7
	COLOR_TP,	//8
	COLOR_DENSE,
	COLOR_CNT
};

enum TIMESTAT
{
	TIME_DYNAMICS,
	TIME_TRANSITION,
	TIME_DISPLAY,
	TIME_TOTAL,
	TIME_COUNT
};

typedef struct AABB
{
	float xMin, xMax;
	float yMin, yMax;
	float zMin, zMax;
} *pAabb;

//0~ total blue, >=6~total red.
__host__ __device__ inline float3 mapColorBlue2Red(float v);

struct matrix4
{
	float m[16];
};

static __inline__ __host__ __device__ matrix3x3 make_matrix3x3(float a, float b, float c, float d, float e, float f, float g, float h, float i) 
{
	matrix3x3 t;
	t.x00 = a; t.x01 = b; t.x02 = c;
	t.x10 = d; t.x11 = e; t.x12 = f;
	t.x20 = g; t.x21 = h; t.x22 = i;
	return t;
}
static __inline__ __host__ __device__ matrix3x3 make_matrix3x3(float a) {
	matrix3x3 t;
	t.x00 = a; t.x01 = a; t.x02 = a; 
	t.x10 = a; t.x11 = a; t.x12 = a;
	t.x20 = a; t.x21 = a; t.x22 = a;
	return t;
}

inline __host__ __device__ void operator+=(matrix3x3 &a, matrix3x3 b) {
	a.x00 += b.x00; a.x01 += b.x01; a.x02 += b.x02;
	a.x10 += b.x10; a.x11 += b.x11; a.x12 += b.x12;
	a.x20 += b.x20; a.x21 += b.x21; a.x22 += b.x22;
}

inline __host__ __device__ matrix3x3 operator+(matrix3x3 A, matrix3x3 B)
	{
		B.x00 += A.x00; B.x01 += A.x01; B.x02 += A.x02;
		B.x10 += A.x10; B.x11 += A.x11; B.x12 += A.x12;
		B.x20 += A.x20; B.x21 += A.x21; B.x22 += A.x22;
		return B;
	}
inline __host__ __device__ matrix3x3 operator*(matrix3x3 B, float b)
{
	matrix3x3 A;
	B.x00 *= b; B.x01 *= b; B.x02 *= b;
	B.x10 *= b; B.x11 *= b; B.x12 *= b;
	B.x20 *= b; B.x21 *= b; B.x22 *= b;
	return B;
}

inline __host__ __device__ matrix3x3 operator*(float b, matrix3x3 B)
{
	matrix3x3 A;
	B.x00 *= b; B.x01 *= b; B.x02 *= b;
	B.x10 *= b; B.x11 *= b; B.x12 *= b;
	B.x20 *= b; B.x21 *= b; B.x22 *= b;
	return B;
}

inline matrix3x3 operator/(matrix3x3 B, float b)
{
	matrix3x3 A;
	B.x00 /= b; B.x01 /= b; B.x02 /= b;
	B.x10 /= b; B.x11 /= b; B.x12 /= b;
	B.x20 /= b; B.x21 /= b; B.x22 /= b;
	
	return B;
}
//moved from spray.cu

inline __host__ __device__ float determinant(matrix3x3 m)
{
	//float result1, result2, result3;
	//result1 = +m.x00 * (m.x11 * m.x22 - m.x21 * m.x12);
	//result2 = -m.x10 * (m.x01 * m.x22 - m.x21 * m.x02);
	//result3 = +m.x20 * (m.x01 * m.x12 - m.x11 * m.x02);
	////printf("result1 = %f, result2 = %f, result3 = %f\n", result1, result2, result3);
	//printf("%f * %f = %f\n", m.x00, m.x11 * m.x22 - m.x21 * m.x12, result1);
	//printf("%f %f %f\n", m.x10, m.x01 * m.x22, m.x21 * m.x02);
	//printf("%f %f %f\n", m.x20, m.x01 * m.x12, m.x11 * m.x02);
	return
		+m.x00 * (m.x11 * m.x22 - m.x21 * m.x12)
		- m.x10 * (m.x01 * m.x22 - m.x21 * m.x02)
		+ m.x20 * (m.x01 * m.x12 - m.x11 * m.x02);
}


inline __host__ __device__ matrix3x3 inverse(matrix3x3 m)
{

	float Determinant = determinant(m);

	matrix3x3 Inverse;

	Inverse.x00 = +(m.x11 * m.x22 - m.x21 * m.x12) / Determinant;
	Inverse.x10 = -(m.x10 * m.x22 - m.x20 * m.x12) / Determinant;
	Inverse.x20 = +(m.x10 * m.x21 - m.x20 * m.x11) / Determinant;
	Inverse.x01 = -(m.x01 * m.x22 - m.x21 * m.x02) / Determinant;
	Inverse.x11 = +(m.x00 * m.x22 - m.x20 * m.x02) / Determinant;
	Inverse.x21 = -(m.x00 * m.x21 - m.x20 * m.x01) / Determinant;
	Inverse.x02 = +(m.x01 * m.x12 - m.x11 * m.x02) / Determinant;
	Inverse.x12 = -(m.x00 * m.x12 - m.x10 * m.x02) / Determinant;
	Inverse.x22 = +(m.x00 * m.x11 - m.x10 * m.x01) / Determinant;

	//Inverse /= Determinant;

	return Inverse;
}


inline __host__ __device__ matrix3x3 mul(float3 a, float3 b)
{
	return make_matrix3x3(
		a.x*b.x, a.x*b.y, a.x*b.z,
		a.y*b.x, a.y*b.y, a.y*b.z,
		a.z*b.x, a.z*b.y, a.z*b.z);
}
inline __host__ __device__ float mul_float(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ matrix3x3 mul(float3 a)
{
	return make_matrix3x3(
		a.x*a.x, a.x*a.y, a.x*a.z,
		a.y*a.x, a.y*a.y, a.y*a.z,
		a.z*a.x, a.z*a.y, a.z*a.z);
}
inline __host__ __device__ matrix3x3 operator*(matrix3x3 a, matrix3x3 b) {
	return make_matrix3x3(
		a.x00*b.x00 + a.x01*b.x10 + a.x02*b.x20,
		a.x00*b.x01 + a.x01*b.x11 + a.x02*b.x21,
		a.x00*b.x02 + a.x01*b.x12 + a.x02*b.x22,

		a.x10*b.x00 + a.x11*b.x10 + a.x12*b.x20,
		a.x10*b.x01 + a.x11*b.x11 + a.x12*b.x21,
		a.x10*b.x02 + a.x11*b.x12 + a.x12*b.x22,

		a.x20*b.x00 + a.x21*b.x10 + a.x22*b.x20,
		a.x20*b.x01 + a.x21*b.x11 + a.x22*b.x21,
		a.x20*b.x02 + a.x21*b.x12 + a.x22*b.x22);
}
//matrix3Mulfloat3
inline __host__ __device__ float3 operator*(matrix3x3 a, float3 b) {
	float3 t;
	//= make_float3(
	t.x = a.x00*b.x + a.x01*b.y + a.x02*b.z,
	t.y = a.x10*b.x + a.x11*b.y + a.x12*b.z,
	t.z = a.x20*b.x + a.x21*b.y + a.x22*b.z;
	return t;
}
//float3Mulmatrix3
inline __host__ __device__ float3 operator*(float3 a, matrix3x3 b) {
	float3 t;
	t.x = a.x*b.x00 + a.y*b.x10 + a.z*b.x20;
	t.y = a.x*b.x01 + a.y*b.x11 + a.z*b.x21;
	t.z = a.x*b.x02 + a.y*b.x12 + a.z*b.x22;
	return t;
}
struct  float9
{
	float x0, x1, x2, x3, x4, x5, x6, x7, x8;
	//float x, y, z, x2, y2, z2, xy, yz, zx;
	/////xu
	float x[9];//
};

struct matrix9x9
{
	float x[9][9];
// 	float x00, x01, x02, x03, x04, x05, x06, x07, x08;
// 	float x10, x11, x12, x13, x14, x15, x16, x17, x18;
// 	float x20, x21, x22, x23, x24, x25, x26, x27, x28;
// 	float x30, x31, x32, x33, x34, x35, x36, x37, x38;
// 	float x40, x41, x42, x43, x44, x45, x46, x47, x48;
// 	float x50, x51, x52, x53, x54, x55, x56, x57, x58;
// 	float x60, x61, x62, x63, x64, x65, x66, x67, x68;
// 	float x70, x71, x72, x73, x74, x75, x76, x77, x78;
// 	float x80, x81, x82, x83, x84, x85, x86, x87, x88;
};

/////xu
struct matrix3x9
{
	float x[3][9];
	//float x00, x01, x02, x03, x04, x05, x06, x07, x08;
	//float x10, x11, x12, x13, x14, x15, x16, x17, x18;
	//float x20, x21, x22, x23, x24, x25, x26, x27, x28;
};

inline matrix3x9 operator+(matrix3x9 A, matrix3x9 B)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 9; j++)
			B.x[i][j] += A.x[i][j];
	}
	return B;
}

inline matrix3x9 operator*(matrix3x9 B, float b)
{
	matrix3x9 A;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 9; j++)
			B.x[i][j] *= b;
	}
	return B;
}

inline matrix3x9 operator/(matrix3x9 B, float b)
{
	matrix3x9 A;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 9; j++)
			B.x[i][j] /= b;
	}
	return B;
}

inline matrix9x9 float9Multfloat9_9x9(float9 x)
{
	matrix9x9 A;
	for (int i = 0; i < 9; i++){
		for (int j = 0; j < 9; j++)
			A.x[i][j] = x.x[i] * x.x[j];
	}
	return A;
}


struct cluster
{
	int num;
	float3 restCM;
	float wsum;
	int snum;
	float3 evedis;
	bool havepar;
	int parnum;
	float3 cm;

	matrix3x3 invRestMat;
	matrix3x3 A;
	float det;
	matrix3x3 mat;
	matrix3x3 Apq;
	matrix3x3 Aqq;
	matrix3x3 R, U, D;
	float detA;

	matrix9x9 A_bar;
	float9 q_bar;//useless
	matrix9x9 Aqq_bar;
	matrix3x9 Apq_bar;
	matrix9x9 invRestMat_bar;
	matrix3x9 mat_bar;
	float det_bar;
	matrix3x9 ApqMultAqq_bar;
	matrix3x9 R_bar;

	//float corr_cluster
};
//

__host__ __device__ float determinant(matrix3x3 m);
/////xu
float determinant(matrix9x9 arcs, int n);
matrix9x9 getAStart(matrix9x9 arcs);
matrix9x9 inverse(matrix9x9 src);
float3 mat39Multfloat9(matrix3x9 a, float9 b);
//
__host__ __device__ matrix3x3 inverse(matrix3x3 m);
__host__ __device__ matrix3x3 BmulD(float3 Bx0, float3 Bx1, float3 Bx2, matrix3x3 D);
matrix3x3 mat3Multmat3(matrix3x3 a, matrix3x3 b);
float3 mat3Multfloat3(matrix3x3 a, float3 b);
matrix3x3 polarDecompositionStable(matrix3x3 mat, float eps);
float3 matRow(matrix3x3 m, int i);
float3 matCol(matrix3x3 m, int i);
matrix3x3 polarDecomposition(matrix3x3 mat, matrix3x3 R, matrix3x3 U, matrix3x3 D);

matrix3x3 make_matrix3x3(float a, float b, float c, float d, float e, float f, float g, float h, float i);
matrix3x3 make_matrix3x3(float a);
matrix3x3 mul(float3 a, float3 b);
matrix3x3 mul(float3 a);
float mul_float(float3 a, float3 b);
#endif
