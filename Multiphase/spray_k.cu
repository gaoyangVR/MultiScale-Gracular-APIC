#include <cuda_runtime.h>    // includes cuda.h and cuda_runtime_api.h
#include "spray_k.cuh"
#include<helper_cuda.h>
#include<helper_math.h>
#include "utility.h"
#include "tables.h"

__constant__ FlipConstant dparam;
__constant__ int NX;
__constant__ int NY;
__constant__ int NZ;
__constant__ int NXMC;
__constant__ int NYMC;
__constant__ int NZMC;
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;

__device__ float racc = 0.;
__device__ float wacc = 0.;
__device__ float3 pacc;
__device__ float sradiusInv;


void copyparamtoGPU(FlipConstant hparam)
{
	checkCudaErrors(cudaMemcpyToSymbol(dparam, &hparam, sizeof(FlipConstant)));
}

void copyNXNYNZtoGPU(int nx, int ny, int nz)
{
	checkCudaErrors(cudaMemcpyToSymbol(NX, &nx, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NY, &ny, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NZ, &nz, sizeof(int)));
}

void copyNXNYNZtoGPU_MC(int nx, int ny, int nz)
{
	checkCudaErrors(cudaMemcpyToSymbol(NXMC, &nx, sizeof(int))); 
	checkCudaErrors(cudaMemcpyToSymbol(NYMC, &ny, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NZMC, &nz, sizeof(int)));
}

__device__ inline void getijk(int &i, int &j, int &k, int &idx)
{
	i = idx / (NZ*NY);
	j = idx / NZ%NY;
	k = idx%NZ;
}

__device__ inline void getijkfrompos(int &i, int &j, int &k, float3 pos)
{
	pos = (pos - dparam.gmin) / dparam.cellsize;
	i = (pos.x >= 0 && pos.x<NX) ? ((int)pos.x) : 0;
	j = (pos.y >= 0 && pos.y<NY) ? ((int)pos.y) : 0;
	k = (pos.z >= 0 && pos.z<NZ) ? ((int)pos.z) : 0;
}
__device__ inline void getijkfrompos(int &i, int &j, int &k, float3 pos, int w, int h, int d, float dx)
{
	pos = (pos - dparam.gmin) / dx;
	i = (pos.x >= 0 && pos.x<w) ? ((int)pos.x) : 0;
	j = (pos.y >= 0 && pos.y<h) ? ((int)pos.y) : 0;
	k = (pos.z >= 0 && pos.z<d) ? ((int)pos.z) : 0;
}

__device__ inline int getidx(int i, int j, int k)
{
	return (i*NZ*NY + j*NZ + k);
}

__device__ inline int getidx(int i, int j, int k, int w, int h, int d)
{
	return (i*h*d + j*d + k);
}

__device__ inline float getRfromMass(float m)
{
	return pow(m*0.75f / M_PI / dparam.waterrho, 0.333333);
}
__device__ inline float getMassfromR(float r)
{
	return dparam.waterrho*M_PI*4.0 / 3 * r*r*r;
}

//
__global__ void cptdivergence(farray outdiv, farray ux, farray uy, farray uz, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		//mark:fluid to spray
		if (mark[idx] == TYPESPRAY)
			div = (ux(i + 1, j, k) - ux(i, j, k) + uy(i, j + 1, k) - uy(i, j, k) + uz(i, j, k + 1) - uz(i, j, k)) / h;

		outdiv[idx] = div;
	}
}

__device__ inline int clampidx(int i, int j, int k)
{
	i = max(0, min(i, NX - 1));
	j = max(0, min(j, NY - 1));
	k = max(0, min(k, NZ - 1));
	return (i*NZ*NY + j*NZ + k);
}



__device__ inline float trilinear(farray u, float x, float y, float z, int w, int h, int d)
{
	x = fmaxf(0.0f, fminf(x, w));
	y = fmaxf(0.0f, fminf(y, h));
	z = fmaxf(0.0f, fminf(z, d));
	int i = fminf(x, w - 2);
	int j = fminf(y, h - 2);
	int k = fminf(z, d - 2);

	return (k + 1 - z)*((j + 1 - y)*((i + 1 - x)*u(i, j, k) + (x - i)*u(i + 1, j, k)) + (y - j)*((i + 1 - x)*u(i, j + 1, k) + (x - i)*u(i + 1, j + 1, k))) +
		(z - k)*((j + 1 - y)*((i + 1 - x)*u(i, j, k + 1) + (x - i)*u(i + 1, j, k + 1)) + (y - j)*((i + 1 - x)*u(i, j + 1, k + 1) + (x - i)*u(i + 1, j + 1, k + 1)));
}

__device__ inline float3 trilinear_cpa(farray u, float x, float y, float z, int w, int h, int d)
{
	x = fmaxf(0.0f, fminf(x, w));
	y = fmaxf(0.0f, fminf(y, h));
	z = fmaxf(0.0f, fminf(z, d));
	int i = fminf(x, w - 2);
	int j = fminf(y, h - 2);
	int k = fminf(z, d - 2);

	float3 cpa = make_float3(0.0f);
	float tx = x - i;
	float ty = y - j;
	float tz = z - k;

	//gradient of the weights N(x), where N is chosen to be the trilinear interpolation kernel.
	
	cpa.x = -(1 - tz)*(1 - ty)*u(i, j, k)
		+ (1 - tz)*(1 - ty)*u(i + 1, j, k)
		- (1 - tz)*(ty)*u(i, j + 1, k)
		+ (1 - tz)*(ty)*u(i + 1, j + 1, k)
		- (tz)*(1 - ty)*u(i, j, k + 1)
		+ (tz)*(1 - ty)*u(i + 1, j, k + 1)
		- (tz)*(ty)*u(i, j + 1, k + 1)
		+ (tz)*(ty)*u(i + 1, j + 1, k + 1);
	cpa.x /= dparam.cellsize.x;
	cpa.y = -(1 - tz)*(1 - tx)*u(i, j, k)
		- (1 - tz)*(tx)*u(i + 1, j, k)
		+ (1 - tz)*(1 - tx)*u(i, j + 1, k)
		+ (1 - tz)*(tx)*u(i + 1, j + 1, k)
		- (tz)*(1 - tx)*u(i, j, k + 1)
		- (tz)*(tx)*u(i + 1, j, k + 1)
		+ (tz)*(1 - tx)*u(i, j + 1, k + 1)
		+ (tz)*(tx)*u(i + 1, j + 1, k + 1);
	cpa.y /= dparam.cellsize.y;
	cpa.z = -(1 - ty)*(1 - tx)*u(i, j, k)
		- (1 - ty)*(tx)*u(i + 1, j, k)
		- (ty)*(1 - tx)*u(i, j + 1, k)
		- (ty)*(tx)*u(i + 1, j + 1, k)
		+ (1 - ty)*(1 - tx)*u(i, j, k + 1)
		+ (1 - ty)*(tx)*u(i + 1, j, k + 1)
		+ (ty)*(1 - tx)*u(i, j + 1, k + 1)
		+ (ty)*(tx)*u(i + 1, j + 1, k + 1);
	cpa.z /= dparam.cellsize.z;

	//cpa = cpa * 1.1;
	return cpa;
}

__device__ inline matrix3x3 trilinear_B(f3array u, float x, float y, float z, int w, int h, int d)
{
	x = fmaxf(0.0f, fminf(x, w));
	y = fmaxf(0.0f, fminf(y, h));
	z = fmaxf(0.0f, fminf(z, d));
	int i = fminf(x, w - 2);
	int j = fminf(y, h - 2);
	int k = fminf(z, d - 2);

	float tx = x - i;
	float ty = y - j;
	float tz = z - k;
	float dx = tx * dparam.cellsize.x;
	float dy = ty * dparam.cellsize.x;
	float dz = tz * dparam.cellsize.x;

	matrix3x3 B;
	B = (1 - tz)*(1 - ty)*(1 - tx)*mul(u(i, j, k), make_float3(-dx, -dy, -dz)) + 
		(1 - tz)*(1 - ty)*(tx)*mul(u(i + 1, j, k), make_float3(dx, -dy, -dz)) +
		(1 - tz)*(ty)*(1 - tx)*mul(u(i, j + 1, k), make_float3(-dx, dy, -dz)) +
		(1 - tz)*(ty)*(tx)*mul(u(i + 1, j + 1, k), make_float3(dx, dy, -dz)) +
		(tz)*(1 - ty)*(1 - tx)*mul(u(i, j, k + 1), make_float3(-dx, -dy, dz)) +
		(tz)*(1 - ty)*(tx)*mul(u(i + 1, j, k + 1), make_float3(dx, -dy, dz)) +
		(tz)*(ty)*(1 - tx)*mul(u(i, j + 1, k + 1), make_float3(-dx, dy, dz)) +
		(tz)*(ty)*(tx)*mul(u(i + 1, j + 1, k + 1), make_float3(dx, dy, dz));
	return B;
}

__device__ inline float3 trilinear_solid(f3array u, float x, float y, float z, int w, int h, int d)
{
	x = fmaxf(0.0f, fminf(x, w));
	y = fmaxf(0.0f, fminf(y, h));
	z = fmaxf(0.0f, fminf(z, d));
	int i = fminf(x, w - 2);
	int j = fminf(y, h - 2);
	int k = fminf(z, d - 2);

	return (k + 1 - z)*((j + 1 - y)*((i + 1 - x)*u(i, j, k) + (x - i)*u(i + 1, j, k)) + (y - j)*((i + 1 - x)*u(i, j + 1, k) + (x - i)*u(i + 1, j + 1, k))) +
		(z - k)*((j + 1 - y)*((i + 1 - x)*u(i, j, k + 1) + (x - i)*u(i + 1, j, k + 1)) + (y - j)*((i + 1 - x)*u(i, j + 1, k + 1) + (x - i)*u(i + 1, j + 1, k + 1)));
}

__device__ float3 getVectorFromGrid(float3 pos, farray phigrax, farray phigray, farray phigraz)
{
	float3 res;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	//注意：ux,uy,uz的存储方式比较特殊(staggered grid)，三维线性插值也要比较小心
	res.x = trilinear(phigrax, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);
	res.y = trilinear(phigray, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);
	res.z = trilinear(phigraz, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);
	return res;
}

__device__ float getScaleFromFrid(float3 pos, farray phi)
{
	float res;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	//注意：ux,uy,uz的存储方式比较特殊(staggered grid)，三维线性插值也要比较小心
	res = trilinear(phi, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);

	return res;
}

//Jacobi iteration: Ax=b
//todo: check this function and maybe get another solver.
__global__ void JacobiIter(farray outp, farray p, farray b, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float resp = 0, h = dparam.cellsize.x;
		float p1, p2, p3, p4, p5, p6;
		float p0 = p[idx];
		int i, j, k;

		//mark:fluid to spray
		if (mark[idx] == TYPESPRAY)
		{
			getijk(i, j, k, idx);
			p1 = (mark(i + 1, j, k) == TYPEBOUNDARY) ? p0 : p(i + 1, j, k);
			p2 = (mark(i, j + 1, k) == TYPEBOUNDARY) ? p0 : p(i, j + 1, k);
			p3 = (mark(i, j, k + 1) == TYPEBOUNDARY) ? p0 : p(i, j, k + 1);
			p4 = (mark(i - 1, j, k) == TYPEBOUNDARY) ? p0 : p(i - 1, j, k);
			p5 = (mark(i, j - 1, k) == TYPEBOUNDARY) ? p0 : p(i, j - 1, k);
			p6 = (mark(i, j, k - 1) == TYPEBOUNDARY) ? p0 : p(i, j, k - 1);

			resp = (p1 + p2 + p3 + p4 + p5 + p6 - h*h*b(i, j, k)) / 6.0f;
		}
		outp[idx] = resp;
	}
}

__global__ void setPressBoundary(farray press)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (i == 0) press[idx] = press(i + 1, j, k);
		if (j == 0) press[idx] = press(i, j + 1, k);
		if (k == 0) press[idx] = press(i, j, k + 1);
		if (i == NX - 1) press[idx] = press(i - 1, j, k);
		if (j == NY - 1) press[idx] = press(i, j - 1, k);
		if (k == NZ - 1) press[idx] = press(i, j, k - 1);
	}
}

//压强与速度的计算
__global__ void subGradPress(farray p, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float h = dparam.cellsize.x;
	if (idx<dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>0 && i<NX)		//look out for this condition
			ux(i, j, k) -= (p(i, j, k) - p(i - 1, j, k)) / h;
	}
	if (idx<dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if (j>0 && j<NY)		//look out for this condition
			uy(i, j, k) -= (p(i, j, k) - p(i, j - 1, k)) / h;
	}
	if (idx<dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if (k>0 && k<NZ)		//look out for this condition
			uz(i, j, k) -= (p(i, j, k) - p(i, j, k - 1)) / h;
	}
}


__device__ float3 getParticleVelFromGrid(float3 pos, farray ux, farray uy, farray uz)
{
	float3 vel;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	//注意：ux,uy,uz的存储方式比较特殊(staggered grid)，三维线性插值也要比较小心
	vel.x = trilinear(ux, x, y - 0.5f, z - 0.5f, NX + 1, NY, NZ);
	vel.y = trilinear(uy, x - 0.5f, y, z - 0.5f, NX, NY + 1, NZ);
	vel.z = trilinear(uz, x - 0.5f, y - 0.5f, z, NX, NY, NZ + 1);
	return vel;
}
//for APIC
__device__ float3 getParticleVelFromGrid_APIC(float3 pos, f3array nodevel)
{
	float3 vel;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	vel = trilinear_solid(nodevel, x, y, z, NX + 1, NY + 1, NZ + 1);
	return vel;
}
__device__ matrix3x3 getParticleMatBFromGrid_APIC(float3 pos, f3array nodevel)
{
	float3 line;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	matrix3x3 matB;
	matB = trilinear_B(nodevel, x, y, z, NX + 1, NY + 1, NZ + 1);

	return matB;
}
__device__ float3 getParticleCpaFromGrid_APIC(int axis, float3 pos, farray ux, farray uy, farray uz)
{
	float3 cpa = make_float3(0.0f);
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	if (axis == 1){
		cpa = trilinear_cpa(ux, x, y - 0.5f, z - 0.5f, NX + 1, NY, NZ);
	}
	else if (axis == 2){
		cpa = trilinear_cpa(uy, x - 0.5f, y, z - 0.5f, NX, NY + 1, NZ);
	}
	else if (axis == 3){
		cpa = trilinear_cpa(uz, x - 0.5f, y - 0.5f, z, NX, NY, NZ + 1);
	}
	return cpa;
}
__global__ void mapvelg2p_APIC_reset(float3 *cpx, float3 *cpy, float3 *cpz, matrix3x3 *B, float3 *ppos, float3 *vel, char *parflag, int pnum)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		vel[idx] = make_float3(0.0f);
		B[idx] = make_matrix3x3(0.0f);
		cpx[idx] = make_float3(0.0f);
		cpy[idx] = make_float3(0.0f);
		cpz[idx] = make_float3(0.0f);
	}
}

__global__ void mapvelg2p_APIC(float3 *cpx, float3 *cpy, float3 *cpz, matrix3x3 *B, float3 *ppos, float3 *vel, char* parflag, int pnum, farray ux, farray uy, farray uz, f3array nodevel)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		float3 gvel = getParticleVelFromGrid(ipos, ux, uy, uz);
		if (parflag[idx] == TYPESOLID){
			gvel = getParticleVelFromGrid_APIC(ipos, nodevel);
			matrix3x3 matB = getParticleMatBFromGrid_APIC(ipos, nodevel);
			B[idx] = matB;
		}
		if ((parflag[idx] == TYPEFLUID)||(parflag[idx] == TYPESPRAY)){
			cpx[idx] = getParticleCpaFromGrid_APIC(1, ipos, ux, uy, uz);
			cpy[idx] = getParticleCpaFromGrid_APIC(2, ipos, ux, uy, uz);
			cpz[idx] = getParticleCpaFromGrid_APIC(3, ipos, ux, uy, uz);

			//if (cpx[idx].x != 0 || cpx[idx].y != 0 || cpx[idx].z != 0)
			//	printf("vel = (%f %f %f)=>(%f %f %f)\ncpx = %f %f %f, cpy = %f %f %f, cpz = %f %f %f\n\n", vel[idx].x, vel[idx].y, vel[idx].z, gvel.x, gvel.y, gvel.z,
			//	cpx[idx].x, cpx[idx].y, cpx[idx].z, cpy[idx].x, cpy[idx].y, cpy[idx].z, cpz[idx].x, cpz[idx].y, cpz[idx].z);
		}
		vel[idx] = gvel;
	}
}

__global__ void mapvelg2p_flip(float3 *ppos, float3 *vel, char* parflag, int pnum, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		float3 gvel = getParticleVelFromGrid(ipos, ux, uy, uz);

		vel[idx] += gvel;
	}
}

__device__ inline float sharp_kernel(float r2, float h)
{
	return fmax(h*h / fmax(r2, 0.0001f) - 1.0f, 0.0f);
}

__global__ void mapvelp2g_slow(float3 *pos, float3 *vel, int pnum, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float w, weight, RE = 1.4, dis2, usum;
	float3 gpos;
	float scale = 1 / dparam.cellsize.x;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = 0;
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int p = 0; p<pnum; p++)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);
			w = sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p].x;
		}
		usum = (weight>0) ? (usum / weight) : 0.0f;
		ux(i, j, k) = usum;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = 0;
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int p = 0; p<pnum; p++)
		{
			dis2 = dot((pos[p] * scale) - gpos, (pos[p] * scale) - gpos);
			w = sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p].y;
		}
		usum = (weight>0) ? (usum / weight) : 0.0f;
		uy(i, j, k) = usum;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = 0;
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int p = 0; p<pnum; p++)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);
			w = sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p].z;
		}
		usum = (weight>0.00001) ? (usum / weight) : 0.0f;
		uz(i, j, k) = usum;
	}
}

__device__ inline bool verifycellidx(int i, int j, int k)
{
	if (i<0 || i>NX - 1 || j<0 || j>NY - 1 || k<0 || k>NZ - 1)
		return false;
	return true;
}
__device__ inline bool verifycellidx(int i, int j, int k, int w, int h, int d)
{
	if (i<0 || i>w - 1 || j<0 || j>h - 1 || k<0 || k>d - 1)
		return false;
	return true;
}

__global__ void addwindforce_k(float3 *pos, float3 *vel, char* parflag, int pnum, float dt, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (wind.z == 0.0f)
			return;
		//切向力
		float a, b, c;
		//float centripetal = 0.02f;
		//float tangential = 0.02f;
		a = pos[idx].x - 0.5;
		b = pos[idx].y - 0.5;
		c = sqrtf(a * a + b * b);
		wind.z = (1.0f - pos[idx].z * pos[idx].z) * wind.z;
		//wind.z = (1.0f - pos[idx].z) * wind.z;
		if (pos[idx].z > 0.75)
			wind.z = 0;
		wind.y = (a / c) * dparam.tangential;
		wind.x = -(b / c) * dparam.tangential;
		//向心力
		//对于离圆心超过一半的粒子往回拉，增大向心力，f=mv^2/r
		if (c > 0.25)
		{
			//float tmpvel2 = vel[idx].x * vel[idx].x + vel[idx].y + vel[idx].y;
			//float tmpcent = tmpvel2 / c * dt;
			//wind.y += -(b / c) * tmpcent;
			//wind.x += -(a / c) * tmpcent;
			wind.y += -(b / c) * dparam.centripetal * 2;
			wind.x += -(a / c) * dparam.centripetal * 2;
		}
		wind.y += -(b / c) * dparam.centripetal;
		wind.x += -(a / c) * dparam.centripetal;


		if (parflag[idx] == TYPEFLUID)
			vel[idx] += wind * dt;
		if (parflag[idx] == TYPESPRAY)
			vel[idx] += wind * dt;
		//if (parflag[idx] == TYPESOLID)
		//	vel[idx] += dt*dparam.gravity;/**0.8*/
		//if (idx == 10)
		//	printf("vel.z = %f, wind acceleration = %f\n",vel[idx].z, dt*wind.z);
	} 
}

__global__ void addgravityforce_k(float3 *vel, char* parflag, int pnum, float dt, int frame)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//if (idx == 10)
		//	printf("vel.z = %f\n", vel[idx].z);
		if (parflag[idx] == TYPEFLUID)
			vel[idx] += dt*dparam.gravity;
		if (parflag[idx] == TYPESPRAY)
			vel[idx] += dt*dparam.gravity;
		if (parflag[idx] == TYPESOLID)
			vel[idx] += dt*dparam.gravity*1.0;/**0.8*/
			{
			//	vel[idx].y= sin(frame/10.0)*8;
			//	vel[idx].z=vel[idx].x=0;
			}
			//if (idx == 10)
		//	printf("vel.z = %f, gravity acceleration = %f\n",vel[idx].z, dt*dparam.gravity.z);
	}
}

__global__ void applygravity2grid(farray ux, farray uy, farray uz, float dt)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//if (idx<pnum)
	//{
	//	if (parflag[idx] == TYPEFLUID)
	//		vel[idx] += dt*dparam.gravity;
	//	if (parflag[idx] == TYPESOLID)
	//		vel[idx] += dt*dparam.gravity*0.7/**0.8*/;
	//}
	if (idx < dparam.gvnum.x)
	{
	}
	if (idx < dparam.gvnum.y)
	{
	}
	if (idx < dparam.gvnum.z)
	{
		int i, j, k;
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		uz(i, j, k) += dt * dparam.gravity.z;// / uz.gmass[getidx(i, j, k)];
	}
}

__global__ void addbuoyancyforce_k(float dheight, float3 *pos, float3 *vel, char* parflag, int pnum, float dt)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (parflag[idx] == TYPEAIR)
			vel[idx] -= dt*dparam.gravity * 1.1f;		//todo:这里的浮力可以小一些，让气泡上的慢一些，视频快一些，水看起来就不太粘了。
		else if (parflag[idx] == TYPEAIRSOLO)
			vel[idx] -= dt*dparam.gravity * 1.1f;
		else if (parflag[idx] == TYPESOLID)
			vel[idx] -= dt*dparam.gravity * 0.9f;
		// 		else if(parflag[idx] == TYPESOLID && pos[idx].z <= dheight)			//	液面下固体粒子受浮力
		// 			vel[idx] -= dt*dparam.gravity * 0.2f;
	}
}

__global__ void addbuoyancyforce_vel(float velMax, float3 *pos, float3 *vel, char* parflag, int pnum, float dt, float buoyanceRateAir, float buoyanceRateSolo)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		float rate = fmax(velMax - vel[idx].z, 0.0f) / velMax;
		if (parflag[idx] == TYPEAIR)
			vel[idx].z -= dt*dparam.gravity.z * rate * buoyanceRateAir;		//todo:这里的浮力可以小一些，让气泡上的慢一些，视频快一些，水看起来就不太粘了。
		else if (parflag[idx] == TYPEAIRSOLO)
			vel[idx].z -= dt*dparam.gravity.z *rate* buoyanceRateSolo;
		else if (parflag[idx] == TYPESOLID)
		vel[idx].z -= dt*dparam.gravity.z * 0.55f;//0.55f;
		//else if(parflag[idx] == TYPESOLID && pos[idx].z <= dheight)			//	液面下固体粒子受浮力
		 		//	vel[idx] -= dt*dparam.gravity * 0.2f;
	}
}

__global__ void advectparticle(float3 *ppos, float3 *pvel, int pnum, farray ux, farray uy, farray uz, float dt,
	char *parflag, VELOCITYMODEL velmode)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//read in
		float3 ipos = ppos[idx], ivel = pvel[idx];
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.samplespace));

		//pos-->grid xyz
		float3 gvel;
		gvel = getParticleVelFromGrid(ipos, ux, uy, uz);

		//vel[idx] += dt*dparam.gravity;
		ipos += gvel*dt;

		if (velmode == CIP)
			ivel = gvel;
		else if (velmode == FLIP)
			ivel = (1 - FLIP_ALPHA)*gvel + FLIP_ALPHA*pvel[idx];

		//check boundary
		ipos.x = fmax(tmin.x, fmin(tmax.x, ipos.x));
		ipos.y = fmax(tmin.y, fmin(tmax.y, ipos.y));
		ipos.z = fmax(tmin.z, ipos.z);
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//write back
		pvel[idx] = ivel;
		ppos[idx] = ipos;
	}
}

__global__ void advectparticle_RK2(float3 *ppos, float3 *pvel, int pnum, farray ux, farray uy, farray uz, float dt,
	char *parflag, VELOCITYMODEL velmode)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//read in
		float3 ipos = ppos[idx], ivel = pvel[idx];
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.samplespace));

		//pos-->grid xyz
		float3 gvel;
		gvel = getParticleVelFromGrid(ipos, ux, uy, uz);

		
		if (velmode == CIP)
			ivel = gvel;
		else if (velmode == FLIP)
			ivel = (1 - FLIP_ALPHA)*gvel + FLIP_ALPHA*pvel[idx];
		ivel = gvel;

		//mid point: x(n+1/2) = x(n) + 0.5*dt*u(xn)
		float3 midpoint = ipos + gvel * dt * 0.5;
		float3 gvelmidpoint = getParticleVelFromGrid(midpoint, ux, uy, uz);
		// x(n+1) = x(n) + dt*u(x+1/2)
		ipos += gvelmidpoint * dt;

		//check cdary
		if (ipos.x <= tmin.x)
			ipos.x = tmin.x, ivel.x = 0.0f;
		if (ipos.y <= tmin.y)
			ipos.y = tmin.y, ivel.y = 0.0f;
		if (ipos.z <= tmin.z)
			ipos.z = tmin.z, ivel.z = 0.0f;

		if (ipos.x >= tmax.x)
			ipos.x = tmax.x, ivel.x = 0.0f;
		if (ipos.y >= tmax.y)
			ipos.y = tmax.y, ivel.y = 0.0f;
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//write back
		if (parflag[idx] != TYPESOLID)
		{
			pvel[idx] = ivel;
			ppos[idx] = ipos;
		}
		else
			pvel[idx] = ivel;
	}
}

__global__ void flipAirVacuum(charray mark)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] == TYPEVACUUM)
			mark[idx] = TYPEAIR;
	}
}
__global__ void markair(charray mark)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		mark[idx] = TYPEAIR;
	}
}


__global__ void markforsmoke(charray mark, farray spraydense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		//mark:fluid to spray
		if(spraydense[idx]>0 )
			mark[idx] = TYPESPRAY;
	}
}

__global__ void markfluid(charray mark, float3 *pos, char *parflag, int pnum, charray sandmark)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		int i, j, k;
		//todo: ???? Should spray particle count??? or should we have a more accurate mark method.
		if (parflag[idx] == TYPEFLUID ||parflag[idx] == TYPESPRAY)
		{
			getijkfrompos(i, j, k, pos[idx]);
			mark(i, j, k) = TYPEFLUID;		//应该是不需要原子操作的，重复写不会有问题
			//if (parflag[idx] == TYPESPRAY)
			//	sandmark(i, j, k) = TYPESPRAY;
		}
	}
}

//判断一下格子里含有的fluid particle的数量，再决定格子的属性
__global__ void markfluid_dense(charray mark, float *parmass, char *parflag, int pnum, uint *gridstart, uint *gridend, int fluidParCntPerGridThres)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int cntfluidsolid = 0, cntair = 0;

		uint start = gridstart[idx];
		uint end = gridend[idx];
		if (start != CELL_UNDEF)
		{
			for (uint p = start; p<end; ++p)
			{
				if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID)
					cntfluidsolid++;
				else if (parflag[p] == TYPEAIR)
					cntair++;
			}
		}
		//dont need acuum here, without bubble function
		if (cntfluidsolid == 0 && cntair == 0)
			mark[idx] = TYPEVACUUM;
		else 
		if (cntfluidsolid>fluidParCntPerGridThres)
			mark[idx] = TYPEFLUID;
		else
			mark[idx] = TYPEAIR;			
	}
}

__global__ void markBoundaryCell(charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
			mark[idx] = TYPEBOUNDARY;
	}
}

__global__ void setgridcolor_k(float* color, ECOLORMODE mode,
	farray p, farray ux, farray uy, farray uz, farray div,
	farray phi, charray mark, farray dense, farray ls, farray tp, float sigma, float temperatureMax, float temperatureMin)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float3 rescolor = make_float3(0.0);
		int cellindex = NY / 2;
		if (mode == COLOR_PRESS)
		{
			if (/*j != cellindex || */p[idx] == 0)
				rescolor = make_float3(0, 0, 1);
			else if (p[idx]>0)
				rescolor = make_float3(0, 1, 0);
			else if (p[idx]<0)
				rescolor = make_float3(1, 0, 0);
			//rescolor = mapColorBlue2Red( 30000*abs(p[idx]) );
		}

		else if (mode == COLOR_UX)
		{
			if (j != cellindex || ux(i + 1, j, k) + ux(i, j, k)<0)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(0.5*abs(ux(i + 1, j, k) + ux(i, j, k)));
		}
		else if (mode == COLOR_UY)
		{
			if (j != cellindex || uy(i, j + 1, k) + uy(i, j, k)<0)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(0.5*abs(uy(i, j + 1, k) + uy(i, j, k)));
		}
		else if (mode == COLOR_UZ)
		{
			if (j != cellindex/*||uz(i,j,k+1)+uz(i,j,k)<0*/)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(5 * abs(uz(i, j, k)));
		}
		else if (mode == COLOR_DIV)
		{
			if (j != cellindex || div[idx] == 0)
				rescolor = make_float3(0, 0, 1);
			else if (div[idx]>0)
				rescolor = make_float3(0, 1, 0);
			else if (div[idx]<0)
				rescolor = make_float3(1, 1, 0);
		}
		else if (mode == COLOR_PHI)
		{
			if (phi[idx]>3 * NX - 1 || j != cellindex)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(0.5f + phi[idx]);
		}
		else if (mode == COLOR_MARK)
		{
			if (j != cellindex)
				rescolor = make_float3(0, 0, 1);
			else
			{
				if (mark[idx] == TYPEAIR)
					rescolor = make_float3(0, 1, 0);
				else if (mark[idx] == TYPEFLUID)
					rescolor = make_float3(1, 0, 0);
				else if (mark[idx] == TYPEVACUUM)
					rescolor = make_float3(1, 1, 0);
				else if (mark[idx] == TYPEBOUNDARY)
					rescolor = make_float3(0, 1, 1);
				else
					rescolor = make_float3(0, 0, 1);
				//rescolor = mapColorBlue2Red( (int)(mark[idx])+1.0f ) ;
			}
		}
		else if (mode == COLOR_LS)
		{
			if (j == cellindex && ls[idx]>0)
				rescolor = mapColorBlue2Red(abs(ls[idx] / dparam.cellsize.x));
			else
				rescolor = make_float3(0, 0, 1);
		}
		else if (mode == COLOR_TP)
		{
			if (j != cellindex || i == 0 || i == NX - 1 || k == 0 || k == NZ - 1)
				rescolor = make_float3(0, 0, 1);
			else
				//	rescolor = mapColorBlue2Red( abs(tp[idx]*dparam.cellsize.x*5/sigma) );

				//rescolor = mapColorBlue2Red( abs(tp[idx]-353)/5.0f );
				rescolor = mapColorBlue2Red((tp[idx] - temperatureMin) / (temperatureMax - temperatureMin)*6.0f);
		}
		else if (mode == COLOR_DENSE)
		{
			//if (j != cellindex)
			//	rescolor = make_float3(0, 0, 1);
			//else
				//rescolor = mapColorBlue2Red(0.0f);
				rescolor = mapColorBlue2Red(3 * dense[idx]);
		}
		color[idx * 3] = rescolor.x;
		color[idx * 3 + 1] = rescolor.y;
		color[idx * 3 + 2] = rescolor.z;
	}
}

__host__ __device__ inline float3 mapColorBlue2Red(float v)
{
	float3 color;
	if (v<0)
		return make_float3(0.0f, 0.0f, 1.0f);

	int ic = (int)v;
	float f = v - ic;
	switch (ic)
	{
	case 0:
	{
			  color.x = 0;
			  color.y = f / 2;
			  color.z = 1;
	}
		break;
	case 1:
	{

			  color.x = 0;
			  color.y = f / 2 + 0.5f;
			  color.z = 1;
	}
		break;
	case 2:
	{
			  color.x = f / 2;
			  color.y = 1;
			  color.z = 1 - f / 2;
	}
		break;
	case 3:
	{
			  color.x = f / 2 + 0.5f;
			  color.y = 1;
			  color.z = 0.5f - f / 2;
	}
		break;
	case 4:
	{
			  color.x = 1;
			  color.y = 1.0f - f / 2;
			  color.z = 0;
	}
		break;
	case 5:
	{
			  color.x = 1;
			  color.y = 0.5f - f / 2;
			  color.z = 0;
	}
		break;
	default:
	{
			   color.x = 1;
			   color.y = 0;
			   color.z = 0;
	}
		break;
	}
	return color;
}

__global__ void initphi(farray phi, charray mark, char typeflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] == typeflag)
			phi[idx] = -0.5;
		else
			phi[idx] = NX * 3;
	}
}

__global__ void initSolidPhi(farray phi, uint *gridstart, uint *gridend, char *pflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		bool flag = false;
		uint start = gridstart[idx];
		if (start != CELL_UNDEF)
		{
			for (; start<gridend[idx]; start++)
			{
				if (pflag[start] == TYPESOLID)
					flag = true;
			}
		}
		if (flag)
			phi[idx] = -0.5f;
		else
			phi[idx] = 3 * NX;
	}
}

__device__ void solvedistance(float a, float b, float c, float &x)
{
	float d = fmin(a, fmin(b, c)) + 1;
	if (d>fmax(a, fmax(b, c)))
	{
		d = (a + b + c + sqrt(3 - (a - b)*(a - b) - (a - c)*(a - c) - (b - c)*(b - c))) / 3;
	}
	if (d<x) x = d;
}
__global__ void sweepphi(farray phi)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float resphi = phi[idx];
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		{
			if (verifycellidx(i + di, j, k) && verifycellidx(i, j + dj, k) && verifycellidx(i, j, k + dk))
				solvedistance(phi(i + di, j, k), phi(i, j + dj, k), phi(i, j, k + dk), resphi);
		}
		phi[idx] = resphi;
	}
}
__global__ void sweepphibytype(farray phi, charray mark, char typeflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] == typeflag)
			return;
		int i, j, k;
		getijk(i, j, k, idx);
		float resphi = phi[idx];
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		{
			if (verifycellidx(i + di, j, k) && verifycellidx(i, j + dj, k) && verifycellidx(i, j, k + dk))
				solvedistance(phi(i + di, j, k), phi(i, j + dj, k), phi(i, j, k + dk), resphi);
		}
		phi[idx] = resphi;
	}
}

__global__ void sweepu(farray outux, farray outuy, farray outuz, farray ux, farray uy, farray uz, farray phi, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	float wx, wy, wz, wsum;		//三个方向上的权重
	if (idx < dparam.gvnum.x)
	{
		//copy
		outux[idx] = ux[idx];

		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>1 && i<NX - 1 /*&& j>0 && j<N-1 && k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i - 1, j, k) == TYPEAIR) || (mark(i, j, k) == TYPEBOUNDARY && mark(i - 1, j, k) == TYPEBOUNDARY))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (j + dj<0 || j + dj>NY - 1 || k + dk<0 || k + dk >NZ - 1)
					continue;
				wx = -di*(phi(i, j, k) - phi(i - 1, j, k));
				if (wx<0)
					continue;
				wy = (phi(i, j, k) + phi(i - 1, j, k) - phi(i, j + dj, k) - phi(i - 1, j + dj, k))*0.5f;
				if (wy<0)
					continue;
				wz = (phi(i, j, k) + phi(i - 1, j, k) - phi(i, j, k + dk) - phi(i - 1, j, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outux(i, j, k) = wx*ux(i + di, j, k) + wy* ux(i, j + dj, k) + wz* ux(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//copy
		outuy[idx] = uy[idx];

		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if ( /*i>0 && i<N-1 &&*/ j>1 && j<NY - 1 /*&& k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j - 1, k) == TYPEAIR) || (mark(i, j, k) == TYPEBOUNDARY && mark(i, j - 1, k) == TYPEBOUNDARY))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di>NX - 1 || k + dk<0 || k + dk >NZ - 1)
					continue;
				wy = -dj*(phi(i, j, k) - phi(i, j - 1, k));
				if (wy<0)
					continue;
				wx = (phi(i, j, k) + phi(i, j - 1, k) - phi(i + di, j, k) - phi(i + di, j - 1, k))*0.5f;
				if (wx<0)
					continue;
				wz = (phi(i, j, k) + phi(i, j - 1, k) - phi(i, j, k + dk) - phi(i, j - 1, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuy(i, j, k) = wx*uy(i + di, j, k) + wy* uy(i, j + dj, k) + wz* uy(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//copy
		outuz[idx] = uz[idx];

		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if ( /*i>0 && i<N-1 && j>0 && j<N-1 &&*/ k>1 && k<NZ - 1)
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j, k - 1) == TYPEAIR) || (mark(i, j, k) == TYPEBOUNDARY && mark(i, j, k - 1) == TYPEBOUNDARY))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di >NX - 1 || j + dj<0 || j + dj>NY - 1)
					continue;
				wz = -dk*(phi(i, j, k) - phi(i, j, k - 1));
				if (wz<0)
					continue;
				wy = (phi(i, j, k) + phi(i, j, k - 1) - phi(i, j + dj, k) - phi(i, j + dj, k - 1))*0.5f;
				if (wy<0)
					continue;
				wx = (phi(i, j, k) + phi(i, j, k - 1) - phi(i + di, j, k) - phi(i + di, j, k - 1))*0.5f;
				if (wx<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuz(i, j, k) = wx*uz(i + di, j, k) + wy* uz(i, j + dj, k) + wz* uz(i, j, k + dk);
			}
		}
	}
}

__global__ void setSmokeBoundaryU_k(farray ux, farray uy, farray uz, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	if (idx < dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		{
			if (i <= 1 || i >= ux.xn - 2)
				ux(i, j, k) = 0.0f;
			else if (j == 0)
				ux(i, j, k) = ux(i, j + 1, k);
			else if (j == NY - 1)
				ux(i, j, k) = ux(i, j - 1, k);
			else if (k == 0)
				ux(i, j, k) = ux(i, j, k + 1);
			else if (k == NZ - 1)
				ux(i, j, k) = ux(i, j, k - 1);
			else if (i>1 && i<NX - 1 && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i - 1, j, k) == TYPEBOUNDARY)))
				ux(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		{
			if (j <= 1 || j >= uy.yn - 2)
				uy(i, j, k) = 0.0f;
			else if (i == 0)
				uy(i, j, k) = uy(i + 1, j, k);
			else if (i == NX - 1)
				uy(i, j, k) = uy(i - 1, j, k);
			else if (k == 0)
				uy(i, j, k) = uy(i, j, k + 1);
			else if (k == NZ - 1)
				uy(i, j, k) = uy(i, j, k - 1);
			else if (j>0 && j<NY && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j - 1, k) == TYPEBOUNDARY)))
				uy(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		{
			if (k <= 1 || k >= uz.zn - 2)
				uz(i, j, k) = 0.0f;
			else if (i == 0)
				uz(i, j, k) = uz(i + 1, j, k);
			else if (i == NX - 1)
				uz(i, j, k) = uz(i - 1, j, k);
			else if (j == 0)
				uz(i, j, k) = uz(i, j + 1, k);
			else if (j == NY - 1)
				uz(i, j, k) = uz(i, j - 1, k);
			else if (k>0 && k<NZ && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j, k - 1) == TYPEBOUNDARY)))
				uz(i, j, k) = 0.0f;
		}
	}
}

__global__ void setWaterBoundaryU_k(farray ux, farray uy, farray uz, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	if (idx < dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		{
			if (i <= 1 || i >= ux.xn - 2)
				ux(i, j, k) = 0.0f;
			else if (i>1 && i<NX - 1 && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i - 1, j, k) == TYPEBOUNDARY)))
				ux(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		{
			if (j <= 1 || j >= uy.yn - 2)
				uy(i, j, k) = 0.0f;
			else if (j>0 && j<NY && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j - 1, k) == TYPEBOUNDARY)))
				uy(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		{
			if (k <= 1 || k >= uz.zn - 1)		//特殊处理ceiling
				uz(i, j, k) = 0.0f;
			else if (k == uz.zn - 2)	//ceiling.
				uz(i, j, k) = (uz(i, j, k - 1)<0) ? (uz(i, j, k - 1)) : 0;
			else if (k>0 && k<NZ && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j, k - 1) == TYPEBOUNDARY)))
				uz(i, j, k) = 0.0f;
		}
	}
}

__global__ void computeDeltaU(farray ux, farray uy, farray uz, farray uxold, farray uyold, farray uzold)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.x)
		uxold[idx] = ux[idx] - uxold[idx];
	if (idx < dparam.gvnum.y)
		uyold[idx] = uy[idx] - uyold[idx];
	if (idx < dparam.gvnum.z)
		uzold[idx] = uz[idx] - uzold[idx];
}


// From CUDA SDK: calculate grid hash value for each particle
__global__ void calcHashD(uint*   gridParticleHash,  // output
	uint*   gridParticleIndex, // output
	float3* pos,               // input: positions
	uint    numParticles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	float3 p = pos[index];

	// get address in grid
	int i, j, k;
	getijkfrompos(i, j, k, p);
	int gridindex = getidx(i, j, k);

	// store grid hash and particle index
	gridParticleHash[index] = gridindex;
	gridParticleIndex[index] = index;
}
// From CUDA SDK: calculate grid hash value for each particle
__global__ void calcHashD_MC(uint*   gridParticleHash,  // output
	uint*   gridParticleIndex, // output
	float3* pos,               // input: positions
	uint    numParticles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;
	
	float3 p = pos[index];

	// get address in grid
	int i, j, k;
	getijkfrompos(i, j, k, p, NXMC, NYMC, NZMC, dparam.cellsize.x / NXMC*NX);
	int gridindex = getidx(i, j, k, NXMC, NYMC, NZMC);

	// store grid hash and particle index
	gridParticleHash[index] = gridindex;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void reorderDataAndFindCellStartD(uint*   cellStart,        // output: cell start index
	uint*   cellEnd,          // output: cell end index
	float3* sortedPos,        // output: sorted positions
	float3* sortedVel,        // output: sorted velocities
	int* sortedRanindex,
	char* sortedflag,
	float* sortedmass,
	float* sortedTemperature,
	float* sortedheat,
	float* sortedsolubility,
	float* sortedgascontain,
	uint *  gridParticleHash, // input: sorted grid hashes
	uint *  gridParticleIndex,// input: sorted particle indices
	float3* oldPos,           // input: sorted position array
	float3* oldVel,           // input: sorted velocity array
	int* oldRanindex,
	char* oldflag,
	float* oldmass,
	float* oldtemperature,
	float* oldheat,
	float* oldsolubility,
	float* oldgascontain,
	uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look 
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		float3 pos = oldPos[sortedIndex];       // macro does either global read or texture fetch
		float3 vel = oldVel[sortedIndex];       // see particles_kernel.cuh
		int ranindex = oldRanindex[sortedIndex];

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedRanindex[index] = ranindex;
		sortedflag[index] = oldflag[sortedIndex];
		sortedmass[index] = oldmass[sortedIndex];
		sortedTemperature[index] = oldtemperature[sortedIndex];
		sortedheat[index] = oldheat[sortedIndex];
		sortedsolubility[index] = oldsolubility[sortedIndex];
		sortedgascontain[index] = oldgascontain[sortedIndex];
	}
}

__global__ void advectux(farray outux, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.x)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx, ux.xn, ux.yn, ux.zn);
		float3 pos = make_float3(i, j + 0.5, k + 0.5);
		//get rid of boundary
		if (i*j*k == 0 || i == NX || j == NY - 1 || k == NZ - 1)
			outux[idx] = 0;
		else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = ux[idx];
			vel.y = (uy(i - 1, j, k) + uy(i - 1, j + 1, k) + uy(i, j, k) + uy(i, j + 1, k))*0.25f;
			vel.z = (uz(i - 1, j, k) + uz(i - 1, j, k + 1) + uz(i, j, k) + uz(i, j, k + 1))*0.25f;

			//if (length(wind) != 0)
			if (0)
			{
				//wind
				float a, b, c;
				a = pos.x * dparam.cellsize.x - 0.5;
				b = pos.y * dparam.cellsize.x - 0.5;
				c = sqrtf(a * a + b * b);
				wind.z = (1.0f - pos.z * dparam.cellsize.x * pos.z * dparam.cellsize.x) * wind.z;
				wind.y = (a / c) * dparam.tangential;
				wind.x = -(b / c) * dparam.tangential;
				//向心力
				//对于离圆心超过一半的粒子往回拉，增大向心力，f=mv^2/r
				if (c > sqrtf(0.125))
				{
					wind.y += -(b / c) * dparam.centripetal * 2;
					wind.x += -(a / c) * dparam.centripetal * 2;
				}
				wind.y += -(b / c) * dparam.centripetal;
				wind.x += -(a / c) * dparam.centripetal;

				vel += wind * 0.01;// * dparam.dt;
			}

			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float oldu = trilinear(ux, oldpos.x, oldpos.y - 0.5f, oldpos.z - 0.5f, ux.xn, ux.yn, ux.zn);
			outux[idx] = oldu * velocitydissipation;
		}
	}
}

__global__ void advectuy(farray outuy, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.y)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx, uy.xn, uy.yn, uy.zn);
		float3 pos = make_float3(i + 0.5, j, k + 0.5);
		//get rid of boundary
		if (i*j*k == 0 || i == NX - 1 || j == NY || k == NZ - 1)
			outuy[idx] = 0;
		else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = (ux(i, j - 1, k) + ux(i + 1, j - 1, k) + ux(i, j, k) + ux(i + 1, j, k))*0.25f;
			vel.y = uy[idx];
			vel.z = (uz(i, j - 1, k) + uz(i, j - 1, k + 1) + uz(i, j, k) + uz(i, j, k + 1))*0.25f;

			//if (length(wind) != 0)
			if (0)
			{
				//wind
				float a, b, c;
				a = pos.x * dparam.cellsize.x - 0.5;
				b = pos.y * dparam.cellsize.x - 0.5;
				c = sqrtf(a * a + b * b);
				wind.z = (1.0f - pos.z * dparam.cellsize.x * pos.z * dparam.cellsize.x) * wind.z;
				wind.y = (a / c) * dparam.tangential;
				wind.x = -(b / c) * dparam.tangential;
				//向心力
				//对于离圆心超过一半的粒子往回拉，增大向心力，f=mv^2/r
				if (c > sqrtf(0.125))
				{
					wind.y += -(b / c) * dparam.centripetal * 2;
					wind.x += -(a / c) * dparam.centripetal * 2;
				}
				wind.y += -(b / c) * dparam.centripetal;
				wind.x += -(a / c) * dparam.centripetal;

				vel += wind * 0.01;// * dparam.dt;
			}

			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float oldu = trilinear(uy, oldpos.x - 0.5f, oldpos.y, oldpos.z - 0.5f, uy.xn, uy.yn, uy.zn);
			outuy[idx] = oldu * velocitydissipation;
		}
	}
}

__global__ void advectuz(farray outuz, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.z)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx, uz.xn, uz.yn, uz.zn);
		float3 pos = make_float3(i + 0.5, j + 0.5, k);
		//get rid of boundary
		if (i*j*k == 0 || i == NX - 1 || j == NY - 1 || k == NZ)
			outuz[idx] = 0;
		else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = (ux(i, j, k - 1) + ux(i + 1, j, k - 1) + ux(i, j, k) + ux(i + 1, j, k))*0.25f;
			vel.y = (uy(i, j, k - 1) + uy(i, j + 1, k - 1) + uy(i, j, k) + uy(i, j + 1, k))*0.25f;
			vel.z = uz[idx];

			//if (length(wind) != 0)
			if (0)
			{
				//wind
				float a, b, c;
				a = pos.x * dparam.cellsize.x - 0.5;
				b = pos.y * dparam.cellsize.x - 0.5;
				c = sqrtf(a * a + b * b);
				wind.z = (1.0f - pos.z * dparam.cellsize.x * pos.z * dparam.cellsize.x) * wind.z;
				wind.y = (a / c) * dparam.tangential;
				wind.x = -(b / c) * dparam.tangential;
				//向心力
				//对于离圆心超过一半的粒子往回拉，增大向心力，f=mv^2/r
				if (c > sqrtf(0.125))
				{
					wind.y += -(b / c) * dparam.centripetal * 2;
					wind.x += -(a / c) * dparam.centripetal * 2;
				}
				wind.y += -(b / c) * dparam.centripetal;
				wind.x += -(a / c) * dparam.centripetal;

				vel += wind * 0.01;// * dparam.dt;
			}
			//vel += dparam.gravity * dparam.dt;
			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float oldu = trilinear(uz, oldpos.x - 0.5f, oldpos.y - 0.5f, oldpos.z, uz.xn, uz.yn, uz.zn);
			//float oldu = -dparam.dt*3.8f;
			outuz[idx] = oldu * velocitydissipation;
		}
	}
}

__global__ void advectscaler(farray outscalar, farray scalar, farray ux, farray uy, farray uz, float densedissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx);
		float3 pos = make_float3(i + 0.5, j + 0.5, k + 0.5);
		//get rid of boundary
		//if ( i == NX - 1 || j == NY - 1 || k == NZ - 1)
		//	outscalar[idx] = 0;
		//else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = (ux(i, j, k) + ux(i + 1, j, k))*0.5f;
			vel.y = (uy(i, j, k) + uy(i, j + 1, k))*0.5f;
			vel.z = (uz(i, j, k) + uz(i, j, k + 1))*0.5f;

			//if (length(wind) != 0)
			if (0)
			{
				if (idx == 10)
					printf("DEBUG:wind force = %f\n", wind.z);
				float a, b, c;
				a = pos.x * dparam.cellsize.x - 0.5;
				b = pos.y * dparam.cellsize.x - 0.5;
				c = sqrtf(a * a + b * b);
				wind.z = (1.0f - pos.z * dparam.cellsize.x * pos.z * dparam.cellsize.x) * wind.z;
				wind.y = (a / c) * dparam.tangential;
				wind.x = -(b / c) * dparam.tangential;
				//向心力
				//对于离圆心超过一半的粒子往回拉，增大向心力，f=mv^2/r
				if (c > 0.25)
				{
					wind.y += -(b / c) * dparam.centripetal * 2;
					wind.x += -(a / c) * dparam.centripetal * 2;
				}
				wind.y += -(b / c) * dparam.centripetal;
				wind.x += -(a / c) * dparam.centripetal;

				//enforce wind as an external velocity field.
				if (idx == 10)
					printf("DEBUG:wind force = %f\n", wind.z);
				vel += wind * 0.01;// *dparam.dt;
				if (idx == 10)
					printf("DEBUG:vel.z = %f, wind force = %f\n", vel.z, wind.z * dparam.dt);
			}

			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float olds = trilinear(scalar, oldpos.x - 0.5f, oldpos.y - 0.5f, oldpos.z - 0.5f, NX, NY, NZ);
			outscalar[idx] = olds;// *densedissipation;
		}
	}
}

__global__ void setsmokedense(farray dense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.z)
	{
		int i, j, k;
		getijk(i, j, k, idx, dense.xn, dense.yn, dense.zn);
		if (i>28 && i<36 && j>28 && j<36 && k>8 && k < 16)
			dense[idx] = dparam.m0*6.0f;
	}
	//if (dense[idx] > 0)
	//	printf("dense[idx] = %f\n", dense[idx]);
}

__global__ void setsmokevel(farray uz, farray dense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.z)
	{
		int i, j, k;
		getijk(i, j, k, idx, uz.xn, uz.yn, uz.zn);
		// 		if( i>20 && i<40 && j>20 && j<40 && k<10 )
		// 			uz[idx] = 4.0f;

		// 		if( k>1 && k<NZ-1 )
		// 			if( dense(i,j,k-1)>0 )
		// 				uz[idx] = 4.0f;

		if (k>1 && k<NZ - 1)
		{
			float alpha = 1000.0f;
			uz(i, j, k) += alpha * dense(i, j, k - 1);
		}
	}
}

__global__ void setsmokevel_nozzle(farray ux, farray dense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.x)
	{
		int i, j, k;
		getijk(i, j, k, idx, ux.xn, ux.yn, ux.zn);
		// 		if( i>20 && i<40 && j>20 && j<40 && k<10 )
		// 			uz[idx] = 4.0f;

		//float alpha = 10000.0f;
		if (i>1 && i<NX - 1)
		if (dense(i - 1, j, k)>0)
			ux[idx] = 8.0f;
		//uz(i,j,k) += alpha * dense(i,j,k-1);
	}
}

surface<void, cudaSurfaceType3D> surfaceWrite;

__global__ void writedens2surface_k(farray dens)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		// 		float4 idens = make_float4( 0.0f );
		// 		if(i>10&&i<50 &&j>10&&j<50&&k>10&&k<50 )
		// 			idens = make_float4( 1.0f );
		float4 idens = make_float4(dens[idx] * 10000);
		surf3Dwrite(idens, surfaceWrite, i*sizeof(float4), j, k);		//why *sizeof(float4)?
	}
}

void writedens2surface(cudaArray* cudaarray, int blocknum, int threadnum, farray dense)
{
	cudaBindSurfaceToArray(surfaceWrite, cudaarray);

	//kernel
	writedens2surface_k << <blocknum, threadnum >> >(dense);
}

__device__ float smooth_kernel(float r2, float h) {
	return fmax(1.0f - r2 / (h*h), 0.0f);
}

__device__ float3 sumcellspring(float3 ipos, float3 *pos, float* pmass, char* parflag, uint *gridstart, uint  *gridend, int gidx, float idiameter)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return make_float3(0.0f);
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dist, w;
	float3 spring = make_float3(0.0f);
	float r = 0;
	float mindis = 1.0f;
	for (uint p = start; p<end; ++p)
	{
		//if( parflag[p]!=TYPESOLID )		//solid粒子也应该对别的粒子产生作用才对
		{
			dist = length(pos[p] - ipos);
			r = idiameter;//+getRfromMass( pmass[p] );
			w = pmass[p] * smooth_kernel(dist*dist, r);
			if (dist>0.1f*idiameter)	//太近会产生非常大的弹力
				spring += w*(ipos - pos[p]) / dist;
			else if (dist >0)
				mindis = (mindis < dist) ? mindis : dist;
		}
	}
	//if (mindis < 1)
	//	printf("min distance between particles: %f \n", mindis);
	return spring;
}

__global__ void correctparticlepos(float3* outpos, float3* ppos, float *pmass, char* parflag, int pnum,
	uint* gridstart, uint *gridend, float correctionspring, float correctionradius, float3 *pepos, float *peradius, int penum)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (parflag[idx] == TYPESOLID /*|| parflag[idx]==TYPESPRAY */|| parflag[idx] == TYPEAIRSOLO)
		{
			outpos[idx] = ppos[idx];
			return;
		}
		//if (ppos[idx].z < dparam.cellsize.x * 2)
		//{
		//	outpos[idx] = ppos[idx];
		//	return;
		//}
		float3 ipos = ppos[idx];
		int i, j, k;
		getijkfrompos(i, j, k, ipos);
		float3 spring = make_float3(0.0f);
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.samplespace));

		float re = correctionradius*dparam.cellsize.x;
		//	float re= getRfromMass( pmass[idx] );
		int lv = 1;
		//	float idiameter = 2*pow(0.75*pmass[idx]/dparam.waterrho/M_PI, 1.0/3);		//注意，应该比实际的半径大，相当于SPH中的核函数半径
		for (int di = -lv; di <= lv; di++) for (int dj = -lv; dj <= lv; dj++) for (int dk = -lv; dk <= lv; dk++)
		{
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				spring += sumcellspring(ipos, ppos, pmass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk), re);
			}
		}

		spring *= correctionspring*re;
		//对于修正位移过大的，上限设为0.3格子
		if (length(dparam.dt*spring)>0.3f*dparam.cellsize.x)
			ipos += dparam.cellsize.x * 0.3f * spring / length(spring);
		else
			ipos += dparam.dt*spring;
		ipos.x = fmax(tmin.x, fmin(tmax.x, ipos.x));
		ipos.y = fmax(tmin.y, fmin(tmax.y, ipos.y));
		ipos.z = fmax(tmin.z, fmin(tmax.z, ipos.z));
		outpos[idx] = ipos;
	}
}

__device__ void sumcelldens(float &phi, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID)
		{
			dis = length(pos[p] - gpos);
			if (phi>dis) phi = dis;
		}
	}
}

//[2012][TVCG]Preserving Fluid Sheets with Adaptively Sampled Anisotropic Particles
__global__ void genWaterDensfield(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NX + 1)*(NY + 1)*(NZ + 1))
	{
		float h = dparam.cellsize.x;
		float phi = 8 * fMCDensity*h;		//from flip3d_vs

		//get position
		int i, j, k;
		getijk(i, j, k, idx, NX + 1, NY + 1, NZ + 1);

		float3 p = make_float3(i, j, k)*h;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				sumcelldens(phi, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
			}
		}
		phi = fMCDensity*h - phi;

		if (i*j*k == 0 || i == NX || j == NY || k == NZ)
			phi = fmin(phi, -0.1f);

		outdens[idx] = phi;
	}
}

__device__ float3 sumcelldens2(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float R, char MCParType)
{
	float3 res = make_float3(0.0f);
	if (gridstart[gidx] == CELL_UNDEF)
		return res;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis, w;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == MCParType)
		{
			dis = length(pos[p] - gpos);
			if (dis<R)
			{
				w = R*R - dis*dis;
				w = w*w*w;
				res += pos[p] * w;
				wsum += w;
			}
		}
	}
	return res;
}


//[2012]【CGF】Parallel Surface Reconstruction for Particle-Based Fluids
__global__ void genWaterDensfield2(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, char MCParType)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		float r = 1.0f*h;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);

		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		int rate = 2;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens2(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h*rate, MCParType);
			}
		}
		if (wsum>0)
		{
			center /= wsum;
			phi = r - length(p - center);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -1000.0f;
		//phi = fmin( phi, -10.0f);

		outdens[idx] = phi;
	}
}

__device__ float3 sumcelldens_Gas(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float R, SCENE scene)
{
	float3 res = make_float3(0.0f);
	if (gridstart[gidx] == CELL_UNDEF)
		return res;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis, w;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEAIR || (parflag[p] == TYPEAIRSOLO && scene != SCENE_INTERACTION))
		{
			dis = length(pos[p] - gpos);
			if (dis<R)
			{
				w = R*R - dis*dis;
				w = w*w*w;
				res += pos[p] * w;
				wsum += w;
			}
		}
	}
	return res;
}


//[2012]【CGF】Parallel Surface Reconstruction for Particle-Based Fluids
__global__ void genWaterDensfield_Gas(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		float r = 0.8f*h;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);

		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		int rate = 2;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens_Gas(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h*rate, scene);
			}
		}
		if (wsum>0)
		{
			center /= wsum;
			phi = r - length(p - center);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -1000.0f;
		//phi = fmin( phi, -10.0f);

		outdens[idx] = phi;
	}
}

__device__ float3 sumcelldens_liquidAndGas(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float R, float sradiusInv, float radius, float racc,float wacc, float3 pacc)
{
	float3 res = make_float3(0.0f);
	if (gridstart[gidx] == CELL_UNDEF)
		return res;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis, w;
	//float r = R / 2.;
	
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEAIR || parflag[p] == TYPEAIRSOLO || parflag[p] == TYPEFLUID)
		{
			dis = length(pos[p] - gpos);
// 			{
// 				float s = dot(pos[p] - gpos, pos[p] - gpos)*sradiusInv;//mantaflow
// 				w = max(0., (1. - s));
// 				wacc += w;
// 				racc += radius * w;
// 				pacc += pos[p] * w;
// 				
// 			}
			if (dis<R)
			{
				w = R*R - dis*dis;
				w = w*w*w;
				res += pos[p] * w;
				wsum += w;
			}
		}
	}
	
	return res;
}


//[2012]【CGF】Parallel Surface Reconstruction for Particle-Based Fluids
__global__ void genWaterDensfield_liquidAndGas(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		//float r = 2.5f*sqrt(3.)*1.01*0.5*h;		//mantaFlow flip03_gen 
		float r = 0.65*h;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);


		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		int rate = 2;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens_liquidAndGas(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h*rate, sradiusInv, r,racc,wacc,pacc);

			}
		}
		if (wsum>0)
		{
			center /= wsum;
			phi = r - length(p - center);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

	//	phi = phiv;		//mantaflow

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -1000.0f;
		//phi = fmin( phi, -10.0f);

		outdens[idx] = phi;
	}
}

__device__ float3  sumcelldens3(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float h, char MCParType)
{
	
}

//[2012]【VRIPHYS】An Efficient Surface Reconstruction Pipeline for Particle-Based Fluids
__global__ void genWaterDensfield_GY(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, char MCParType, float3 centertmp)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		float r = 0.75f*h;
		float thigh = 0.51;
		float tlow = 0.49;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);

		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens3(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h, MCParType);
			}
		}

		if (wsum>0)
		{
			center /= wsum;			//~v

			float3 delta = center - centertmp;
			float Ev = max(delta.x, max(delta.y, delta.z)) / (4 * h);	//
			//	float Ev = 3.8;
			centertmp = center;             //	centertmp:存储的是上一次的center 求Ev的delta用
			float gamma = (thigh - Ev) / (thigh - tlow);
			float f = (Ev<tlow) ? 1 : gamma*gamma*gamma - 3 * gamma*gamma + 3 * gamma;

			//		phi = r - length( p - center );
			phi = (length(p - center) - r*f);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = fmin(phi, -10.0f);

		outdens[idx] = phi;
	}
}

__global__ void markSolid_sphere(float3 spherepos, float sphereradius, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if ((i>NX/2-2) &&i<2.5*NX/3 && j>3.5*NY/9 && j< 6*NY/9 && k<NZ/5)
		mark[idx] = TYPEBOUNDARY;
	}
}

__global__ void markSolid_waterfall(int3 minpos, int3 maxpos, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int x, y, z;
		getijk(x, y, z, idx);
		if (x <= maxpos.x && (y >= maxpos.y || y <= minpos.y) && z <= maxpos.z)
			mark[idx] = TYPEBOUNDARY;
		else if (x <= maxpos.x && (y>minpos.y || y<maxpos.y) && z <= minpos.z)
			mark[idx] = TYPEBOUNDARY;
	}
}

//a trick part.
__global__ void markSolid_waterfall_liquid(int3 minpos, int3 maxpos, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int x, y, z;
		getijk(x, y, z, idx);
		if (x <= maxpos.x && (y >= maxpos.y || y <= minpos.y) && z <= maxpos.z*0.7f)
			mark[idx] = TYPEBOUNDARY;
		else if (x <= maxpos.x && (y>minpos.y || y<maxpos.y) && z <= minpos.z*0.7f)
			mark[idx] = TYPEBOUNDARY;
	}
}

//a trick part.
__global__ void markSolid_terrain(charray mark, charray mark_terrain)
{
	
}

__global__ void updatesprayvel_k(farray sprayux, farray sprayuy, farray sprayuz, farray dense,
	farray waterux, farray wateruy, farray wateruz, charray mark, float densegravityMul, float3 wind)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	if (idx < dparam.gvnum.x)
	{
		//x direction
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i != 0 && i != NX)
		{
			
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//y direction
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if (j != 0 && j != NY)
		{
		
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//z direction
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if (k != 0 && k != NZ)
		{
			//if (mark(i, j, k) == TYPEFLUID || mark(i, j, k - 1) == TYPEFLUID)
			//	sprayuz[idx] = wateruz[idx];


			//if (wind.z == 0)
			if (dense(i, j, k) >0)
				sprayuz[idx] += dparam.dt * dparam.gravity.z *densegravityMul;
			
		}

	}
}

__global__ void markwaterparticle_par(float3* ppos, float3* pvel, float* pmass, char* pflag, bool* pparstaymark, int pnum,
	farray ux, farray uy, farray uz, charray mark, uint *transmark, float genSprayRelVelThres, float genSprayVelThres)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < pnum)
	{
		float3 ipos = ppos[idx];
		float3 gvel, ivel = pvel[idx];
		float imass = pmass[idx];
		char iflag = pflag[idx];
		bool iparstaymark = pparstaymark[idx];
		float velthreshold = genSprayRelVelThres;
		float velthreshold2 = genSprayVelThres;

		//condition 0: this is a liquid particle
		if (iflag != TYPEFLUID)
			return;
		//condition 1: near air.
		int i, j, k;
		getijkfrompos(i, j, k, ipos);
		if (mark(i, j, k) == TYPEBOUNDARY)	//safe check
			return;
		//DEBUG
		if (mark(i - 1, j, k) != TYPEFLUID || mark(i + 1, j, k) != TYPEFLUID || mark(i, j - 1, k) != TYPEFLUID || mark(i, j + 1, k) != TYPEFLUID || mark(i, j, k - 1) != TYPEFLUID || mark(i, j, k + 1) != TYPEFLUID)
		{
			//condition 1: relative velocity between particle and grid > thres1
			//condition 2: absolute velocity >thres2
			gvel = getParticleVelFromGrid(ipos, ux, uy, uz);
			if ((length(gvel - ivel) > velthreshold) || (length(ivel) > velthreshold2))
			{
				//if (length(ivel) > velthreshold2)
				{
					//check mass, decide if this particle has been splitted.
					//if ((imass > dparam.m0 - 0.0001) && (!iparstaymark))
					if (imass == dparam.m0)
					{
						//mark
						transmark[idx] = 1;
						//pflag[idx] = TYPESPRAY;		//no split, just change flag.
					}
				}
			}
		}
	}
}


__global__ void Transparticle_Spray2Sand(float3* ppos, float3* pvel, float* pmass, char* pflag, int pnum,
	farray ux, farray uy, farray uz, charray mark, uint *transmark, float genSprayRelVelThres, float genSprayVelThres, farray spraydense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		float3 ipos = ppos[idx];
		float3 gvel, ivel = pvel[idx];
		float imass = pmass[idx];
		char iflag = pflag[idx];
		float velthreshold = genSprayRelVelThres;
		float velthreshold2 = genSprayVelThres;

		//condition 0: this is a spary particle
		if (iflag != TYPESPRAY)
			return;
		
		int i, j, k;
		getijkfrompos(i, j, k, ipos);
		if (mark(i, j, k) == TYPEBOUNDARY)	//safe check
			return;
		////condition 1: dense here above a threshold
		int cellidx = getidx(i, j, k);
		//if (spraydense[cellidx] < dparam.m0 * 36 / 64)
		//	return;
		//condition 2: relative velocity between particle and grid.
		gvel = getParticleVelFromGrid(ipos, ux, uy, uz);
		if (length(gvel - ivel) < velthreshold)
		{
			if (length(ivel) < velthreshold2 * 1.5)
			{
				if (spraydense[cellidx] > dparam.m0 * 36 / 64)
				{
					transmark[idx] = 2;
					pflag[idx] = TYPEFLUID;
					
					spraydense[cellidx] -= dparam.m0 * 36 / 64;
				}
				else if ((k <= 3)&&(spraydense[cellidx] > dparam.m0 * 16 / 64))
				{
						pflag[idx] = TYPEFLUID;
						spraydense[cellidx] == 0;
						pmass[idx] *= 0.5;
					
				}
			}
		}
	}
}

__global__ void MinusSprayDensityFromParticle(farray ux, farray uy, farray uz, farray spraydense,
	uint *gridstart, uint *gridend, charray mark, farray massadd,
	float3* ppos, float3* pvel, float *pwsum, uint *transmark, float* transmass, uint* minusmark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		massadd[idx] = 0.0f;
		if (mark[idx] == TYPEBOUNDARY)
			return;

		int gidx;
		float RE = 2.0f;	// RE should larger than sqrt(3).
		float3 offset = make_float3(0.5f);
		float3 possub, momentumsum = make_float3(0.0f);
		float iw, masssum = 0.0f;
		//for write back.
		float mg;
		float3 vg, gridpos, deltavel;
		vg.x = 0.5f* (ux(i + 1, j, k) + ux(i, j, k));
		vg.y = 0.5f* (uy(i, j + 1, k) + uy(i, j, k));
		vg.z = 0.5f* (uz(i, j, k + 1) + uz(i, j, k));
		mg = spraydense[idx];
		gridpos = make_float3(i, j, k) + offset;

		for (int di = i - 1; di <= i + 1; di++) for (int dj = j - 1; dj <= j + 1; dj++) for (int dk = k - 1; dk <= k + 1; dk++)
		{
			if (verifycellidx(di, dj, dk))
			{
				//all the particles in cell
				gidx = getidx(di, dj, dk);
				uint start = gridstart[gidx];
				if (start == CELL_UNDEF)
					continue;
				uint end = gridend[gidx];
				for (uint p = start; p<end; ++p)
				{
					if (transmark[p] != 2)
						continue;
					// particle transference.
					possub = ppos[p] / dparam.cellsize.x - gridpos;

					//todo: check particle position.
					if (pwsum[p] <= 0.0f)
						continue;
					if (abs(possub.x)>1.0f || abs(possub.y)>1.0f || abs(possub.z)>1.0f)	//否则这个粒子的质量影响不到这个cell上。
						continue;

					//todo: check, this is not quite right.
					iw = sharp_kernel(dot(possub, possub), RE) / pwsum[p];	//这是归一化之后的权值
					masssum += iw * transmass[p];
					momentumsum += iw * transmass[p] * pvel[p];
				}
			}
		}
		if (masssum > 0.0f)	//in case of dividing 0.
		{
			// add mass, update velocity
			deltavel = (mg * vg + momentumsum) / (mg + masssum) - vg;
			spraydense[idx] -= masssum;
			massadd[idx] = masssum;
			
		}
	}
}

__global__ void markDenseTransFromLiquid(float* transmass, uint *densetransmark, int pnum, uint *partransmark, float fWater2Spray)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (partransmark[idx] == 1)
		{
			densetransmark[idx] = 1;
			transmass[idx] = dparam.m0 * 64 / 64 * fWater2Spray;	//todo: this is related to the applied splitting scheme.
		}
		else if (partransmark[idx] == 2)
		{
			densetransmark[idx] = 2;
			transmass[idx] = dparam.m0 * 36 / 64 * fWater2Spray;
		}
	}
}

//changed to delete particles that generate smoke
__global__ void genSprayParticles(int activeparticle, int splitnum, int parnum, uint *compactParticles,
	float3* pos, float3 *vel, char* parflag, float *parmass, float3 *parrandom, int *parranindex, farray ux, farray uy, farray uz,
	float *randfloat, int randfloatcnt)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	
	if (idx < activeparticle)
	{
		int writeindex = parnum - 1 - idx;	//
		int particleindex = compactParticles[idx];
		if (particleindex >= parnum)
			return;		//todo: error.
		pos[particleindex] = pos[writeindex];
		vel[particleindex] = vel[writeindex];
		parflag[particleindex] = parflag[writeindex];
		parmass[particleindex] = parmass[writeindex];
		//switch index
		int temp = 0;
		temp = parranindex[particleindex];
		parranindex[particleindex] = parranindex[writeindex];
		parranindex[writeindex] = temp;
		
		pos[writeindex] = make_float3(0.0f);
		vel[writeindex] = make_float3(0.0f);
		parflag[writeindex] = TYPEFLUID;
		parmass[writeindex] = 0;
		parrandom[parranindex[writeindex]] = make_float3(0.0f);
	}
}

__global__ void Markcelltrans(farray ux, farray uy, farray uz, farray spraydense, uint* celltransmark, uint* denseminusmark, float genSandVelThres, float notgenSandvelThres)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		//get pos of grid point
		int i, j, k;
		getijk(i, j, k, idx);
		
		if (spraydense[idx] < dparam.m0 * 2 / 64)//sandstorm:.. landslide:2/64
			return;

		float vel = sqrtf(ux[idx] * ux[idx] + uy[idx] * uy[idx] + uz[idx] * uz[idx]);
		float vel_xy = sqrtf(ux[idx] * ux[idx] + uy[idx] * uy[idx]);

		//high density generate type 1
		if ((spraydense[idx]> dparam.m0 * 64 / 64) && (vel < notgenSandvelThres))
		{
			celltransmark[idx] = 1;
			denseminusmark[idx] = 1;
		}

		//low speed generate type 2
		else if (vel_xy < genSandVelThres)
		{
			//printf("%f\n\n", vel);
			celltransmark[idx] = 1;
			denseminusmark[idx] = 2;
		}
		//near ground
		//if ((k < 12) && (vel_xy < notgenSandvelThres))
		//{
		//	celltransmark[idx] = 1;
		//	denseminusmark[idx] = 2;
		//}
	}
}

__global__ void genSandParticles(int activecell, int splitnum, int parnum, uint *compactcells, 
	float3* pos, float3 *vel, char* parflag, float *parmass, bool *parstaymark, farray ux, farray uy, farray uz,
	float *randfloat, int randfloatcnt, int frame, farray spraydense, uint* denseminusmark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < activecell)
	{
		int writeindex = parnum + idx;	//向数组末尾追写FLUID粒子
		int i, j, k;
		int cellidx = compactcells[idx];
		//printf("\n%d\n", cellidx);
		getijk(i, j, k, cellidx);
		float3 ipos = make_float3(i + 0.5, j + 0.5, k + 0.5);
		ipos *= dparam.cellsize.x;
		//插值出速度来
		//float3 ivel = getParticleVelFromGrid(ipos, ux, uy, uz);
		float3 ivel;
		ivel.x = 0.5f* (ux(i + 1, j, k) + ux(i, j, k));
		ivel.y = 0.5f* (uy(i, j + 1, k) + uy(i, j, k));
		ivel.z = 0.5f* (uz(i, j, k + 1) + uz(i, j, k));
		//rand number preparation.
		int userandnum = splitnum * 3;
		int randbase = idx % (randfloatcnt - userandnum);
		float radius = 0.5f * dparam.cellsize.x;
		float imass = dparam.m0;

		
		float3 randpos = make_float3(randfloat[randbase], randfloat[randbase + 1], randfloat[randbase + 2]) *2.0f - 1.0f;
		
		//pos[writeindex] = ipos + radius * randpos;
		//Sandstrom dont need random
		pos[writeindex] = ipos;
		vel[writeindex] = ivel;
		parflag[writeindex] = TYPEFLUID;
		parmass[writeindex] = 0.95 * imass;
		parstaymark[writeindex] = true;
		//printf("\nidx = %d, i j k = %d, %d, %d, ipos = %f, %f, %f\n", idx, i, j, k, ipos.x, ipos.y, ipos.z);
		if (denseminusmark[cellidx] == 1)
			spraydense[cellidx] -= imass;
		else if (denseminusmark[cellidx] == 2)
		{
			parmass[writeindex] = spraydense[cellidx];
			spraydense[cellidx] = 0;
			parflag[writeindex] = TYPESPRAY;
			//printf("spraydense[%d] = %f", cellidx, spraydense[cellidx]);
		}
		//消除特殊部位spray粒子速度向上
		vel[writeindex].z = (vel[writeindex].z > 0) ? 0 : vel[writeindex].z;
	}
}

//小trick，在画图测量粒子大小的时候，用下面这个函数，但是它是不保质量的
__global__ void genSprayParticles_radius(int activeparticle, int splitnum, int parnum, uint *compactParticles,
	float3* pos, float3 *vel, char* parflag, float *parmass, farray ux, farray uy, farray uz,
	float *randfloat, int randfloatcnt)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < activeparticle)
	{
		int writeindex = parnum + (splitnum - 1)*idx;	//water particle本身改成一个spray particle,再向数组末尾追写splitnum-1个
		int particleindex = compactParticles[idx];
		if (particleindex >= parnum)
			return;		//todo: error.
		float3 ipos = pos[particleindex];
		//插值出速度来
		float3 ivel = getParticleVelFromGrid(ipos, ux, uy, uz);
		//rand number preparation.
		int userandnum = splitnum * 4;
		int randbase = idx % (randfloatcnt - userandnum);
		float radius = 0.5f*dparam.cellsize.x;
		//	float massrate;

		float imass = parmass[particleindex];
		pos[particleindex] = ipos + radius*make_float3(2 * randfloat[randbase] - 1, 2 * randfloat[randbase + 1] - 1, 2 * randfloat[randbase + 2] - 1);
		randbase += 4;
		//massrate = (powf(randfloat[randbase++]*2-1, 3.0f)+1)*0.5f;
		vel[particleindex] = ivel;
		parflag[particleindex] = TYPESPRAY;
		parmass[particleindex] = imass * 27 / 64;	//3/4*diameter = 27/64*mass.
		for (int j = 0; j<splitnum - 2; j++)
		{
			//todo: distribute
			pos[writeindex + j] = ipos + radius*make_float3(2 * randfloat[randbase] - 1, 2 * randfloat[randbase + 1] - 1, 2 * randfloat[randbase + 2] - 1);
			randbase += 3;
			//massrate = (powf(randfloat[randbase++]*2-1, 3.0f)+1)*0.5f;
			vel[writeindex + j] = ivel;
			parflag[writeindex + j] = TYPESPRAY;
			parmass[writeindex + j] = imass*0.125f*(randfloat[randbase++] + 0.5f);		//1/8*mass
		}
		pos[writeindex + splitnum - 2] = ipos + radius*make_float3(2 * randfloat[randbase] - 1, 2 * randfloat[randbase + 1] - 1, 2 * randfloat[randbase + 2] - 1);
		randbase += 4;
		//massrate = (powf(randfloat[randbase++]*2-1, 3.0f)+1)*0.5f;
		vel[writeindex + splitnum - 2] = ivel;
		parflag[writeindex + splitnum - 2] = TYPESPRAY;
		parmass[writeindex + splitnum - 2] = imass / 64;		//1/64*mass
	}
}
//得到网格上每一个结点的密度值，为MC算法做准备
__global__ void genSphereDensfield(farray outdens, float3 center, float radius)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		//float3 center = make_float3(0.5f);
		float phi;

		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);
		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -0.1;
		else
		{
			float3 p = make_float3(i, j, k)*dparam.cellsize.x / (NXMC / NX);
			phi = radius - length(p - center);
		}
		outdens[idx] = phi;
	}
}

__global__ void sumweight(float3* ppos, uint* transmark, float *pwsum, int pnum, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//本粒子是不是要转换
		if (transmark[idx] == 0)
			return;
		float3 ipos = ppos[idx];
		float3 offset = 0.5f * dparam.cellsize;
		int i, j, k;
		float wsum = 0.0f;
		float dis;
		float RE = 2.0f;	// RE should larger than sqrt(3).
		getijkfrompos(i, j, k, ipos - offset);	
		for (int di = i; di <= i + 1; di++) for (int dj = j; dj <= j + 1; dj++) for (int dk = k; dk <= k + 1; dk++)
			if (verifycellidx(di, dj, dk))
				if (mark(di, dj, dk) != TYPEBOUNDARY)
				{
					//all scaled with delta_x.
					dis = length(make_float3(di, dj, dk) + (offset - ipos) / dparam.cellsize.x);
					wsum += sharp_kernel(dis*dis, RE);		//scale
				}
		pwsum[idx] = wsum;
	}
}

//以cell为核心，计算影响此cell的所有粒子的加权质量和，以及动量的影响。
__global__ void updateSprayDensityFromParticle_k(farray ux, farray uy, farray uz, farray spraydense,
	uint *gridstart, uint *gridend, charray mark, farray massadd,
	float3* ppos, float3* pvel, float *pwsum, uint *transmark, float* transmass)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		massadd[idx] = 0.0f;
		if (mark[idx] == TYPEBOUNDARY)
			return;

		int gidx;
		float RE = 2.0f;	// RE should larger than sqrt(3).
		float3 offset = make_float3(0.5f);
		float3 possub, momentumsum = make_float3(0.0f);
		float iw, masssum = 0.0f;
		//for write back.
		float mg;
		float3 vg, gridpos, deltavel;
		vg.x = 0.5f* (ux(i + 1, j, k) + ux(i, j, k));
		vg.y = 0.5f* (uy(i, j + 1, k) + uy(i, j, k));
		vg.z = 0.5f* (uz(i, j, k + 1) + uz(i, j, k));
		mg = spraydense[idx];
		gridpos = make_float3(i, j, k) + offset;

		for (int di = i - 1; di <= i + 1; di++) for (int dj = j - 1; dj <= j + 1; dj++) for (int dk = k - 1; dk <= k + 1; dk++)
		{
			if (verifycellidx(di, dj, dk))
			{
				//all the particles in cell
				gidx = getidx(di, dj, dk);
				uint start = gridstart[gidx];
				if (start == CELL_UNDEF)
					continue;
				uint end = gridend[gidx];
				for (uint p = start; p<end; ++p)
				{
					if (transmark[p] == 0)
						continue;
					// particle transference.
					possub = ppos[p] / dparam.cellsize.x - gridpos;

					//todo: check particle position.
					if (pwsum[p] <= 0.0f)
						continue;
					if (abs(possub.x)>1.0f || abs(possub.y)>1.0f || abs(possub.z)>1.0f)	//否则这个粒子的质量影响不到这个cell上。
						continue;

					//todo: check, this is not quite right.
					iw = sharp_kernel(dot(possub, possub), RE) / pwsum[p];	//这是归一化之后的权值
					masssum += iw * transmass[p];
					momentumsum += iw * transmass[p] * pvel[p];
				}
			}
		}
		if (masssum > 0.0f)	//in case of dividing 0.
		{
			// add mass, update velocity
			deltavel = (mg * vg + momentumsum) / (mg + masssum) - vg;
			spraydense[idx] += masssum;
			massadd[idx] = masssum;
			atomicAdd(&(ux.data[getidx(i, j, k, NX + 1, NY, NZ)]), deltavel.x*0.5f);
			atomicAdd(&(ux.data[getidx(i + 1, j, k, NX + 1, NY, NZ)]), deltavel.x*0.5f);
			atomicAdd(&(uy.data[getidx(i, j, k, NX, NY + 1, NZ)]), deltavel.y*0.5f);
			atomicAdd(&(uy.data[getidx(i, j + 1, k, NX, NY + 1, NZ)]), deltavel.y*0.5f);
			atomicAdd(&(uz.data[getidx(i, j, k, NX, NY, NZ + 1)]), deltavel.z*0.5f);
			atomicAdd(&(uz.data[getidx(i, j, k + 1, NX, NY, NZ + 1)]), deltavel.z*0.5f);
		}
	}
}

__global__ void updatespraydensity_k(charray mark, farray waterux, farray wateruy, farray wateruz,
	farray dens, farray sprayux, farray sprayuy, farray sprayuz, farray div)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		if (mark[idx] != TYPEFLUID)
			return;
		int i, j, k;
		getijk(i, j, k, idx);
		float vthreshold = 0.5f;
		float densparam = 0.3f;
		float vchangeparam = 0.3f;
		float ulen;
		float divthreshold = 3.0f;
		if (div[idx] < divthreshold)
			return;
		//x方向
		ulen = abs((waterux[idx] + waterux(i + 1, j, k))*0.5f);
		if (ulen>vthreshold)
		{
			if (mark(i, j + 1, k) == TYPEAIR)//y+
			{
				dens(i, j + 1, k) += densparam*ulen;
				sprayuy(i, j + 1, k) += vchangeparam*ulen;
			}
			if (mark(i, j - 1, k) == TYPEAIR)//y-
			{
				dens(i, j - 1, k) += densparam*ulen;
				sprayuy(i, j, k) -= vchangeparam*ulen;
			}
			if (mark(i, j, k + 1) == TYPEAIR)//z+
			{
				dens(i, j, k + 1) += densparam*ulen;
				sprayuz(i, j, k + 1) += vchangeparam*ulen;
			}
			if (mark(i, j, k - 1) == TYPEAIR)//z-
			{
				dens(i, j, k - 1) += densparam*ulen;
				sprayuz(i, j, k) -= vchangeparam*ulen;
			}
		}
		//y方向
		ulen = abs((wateruy[idx] + wateruy(i, j + 1, k))*0.5f);
		if (ulen>vthreshold)
		{
			if (mark(i + 1, j, k) == TYPEAIR)//x+
			{
				dens(i + 1, j, k) += densparam*ulen;
				sprayux(i + 1, j, k) += vchangeparam*ulen;
			}
			if (mark(i - 1, j, k) == TYPEAIR)//x-
			{
				dens(i - 1, j, k) += densparam*ulen;
				sprayux(i, j, k) -= vchangeparam*ulen;
			}
			if (mark(i, j, k + 1) == TYPEAIR)//z+
			{
				dens(i, j, k + 1) += densparam*ulen;
				sprayuz(i, j, k + 1) += vchangeparam*ulen;
			}
			if (mark(i, j, k - 1) == TYPEAIR) //z-
			{
				dens(i, j, k - 1) += densparam*ulen;
				sprayuz(i, j, k) -= vchangeparam*ulen;
			}
		}
		//z方向
		ulen = abs((wateruz[idx] + wateruz(i, j, k + 1))*0.5f);
		if (ulen>vthreshold)
		{
			if (mark(i + 1, j, k) == TYPEAIR)	//x+
			{
				dens(i + 1, j, k) += densparam*ulen;
				sprayux(i + 1, j, k) += vchangeparam*ulen;
			}
			if (mark(i - 1, j, k) == TYPEAIR)	//x-
			{
				dens(i - 1, j, k) += densparam*ulen;
				sprayux(i, j, k) -= vchangeparam*ulen;
			}
			if (mark(i, j + 1, k) == TYPEAIR)	//y+
			{
				dens(i, j + 1, k) += densparam*ulen;
				sprayuy(i, j + 1, k) += vchangeparam*ulen;
			}
			if (mark(i, j - 1, k) == TYPEAIR)	//y-
			{
				dens(i, j - 1, k) += densparam*ulen;
				sprayuy(i, j, k) -= vchangeparam*ulen;
			}
		}
	}
}

//from cuda sdk 4.2
// classify voxel based on number of vertices it will generate
// one thread per voxel (cell)
__global__ void classifyVoxel(uint* voxelVerts, uint *voxelOccupied, farray volume, float isoValue)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<NXMC*NYMC*NZMC)
	{
		int i, j, k;
		getijk(i, j, k, idx, NXMC, NYMC, NZMC);

		float field[8];
		field[0] = volume(i, j, k);
		field[1] = volume(i + 1, j, k);
		field[2] = volume(i + 1, j + 1, k);
		field[3] = volume(i, j + 1, k);
		field[4] = volume(i, j, k + 1);
		field[5] = volume(i + 1, j, k + 1);
		field[6] = volume(i + 1, j + 1, k + 1);
		field[7] = volume(i, j + 1, k + 1);

		// calculate flag indicating if each vertex is inside or outside isosurface
		uint cubeindex;
		cubeindex = uint(field[0] < isoValue);
		cubeindex += uint(field[1] < isoValue) * 2;
		cubeindex += uint(field[2] < isoValue) * 4;
		cubeindex += uint(field[3] < isoValue) * 8;
		cubeindex += uint(field[4] < isoValue) * 16;
		cubeindex += uint(field[5] < isoValue) * 32;
		cubeindex += uint(field[6] < isoValue) * 64;
		cubeindex += uint(field[7] < isoValue) * 128;

		// read number of vertices from texture
		uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

		voxelVerts[idx] = numVerts;
		voxelOccupied[idx] = (numVerts > 0);
	}//endif
}

// compact voxel array
__global__ void
compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (voxelOccupied[i] && (i < numVoxels)) {
		compactedVoxelArray[voxelOccupiedScan[i]] = i;
	}
}

// compute interpolated vertex along an edge
__device__
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

// calculate triangle normal
__device__
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;
	// note - it's faster to perform normalization in vertex shader rather than here
	return cross(edge0, edge1);
}


__device__ int GetVertexID(int i, int j, int k)
{
	return 3 * (i*(NZMC + 1)*(NYMC + 1) + j*(NZMC + 1) + k);
}

__device__ int GetEdgeID(int nX, int nY, int nZ, int edge)
{
	//	return GetVertexID( nX,nY,nZ );

	switch (edge) {
	case 0:
		return GetVertexID(nX, nY, nZ) + 1;
	case 1:
		return GetVertexID(nX + 1, nY, nZ);
	case 2:
		return GetVertexID(nX, nY + 1, nZ) + 1;
	case 3:
		return GetVertexID(nX, nY, nZ);
	case 4:
		return GetVertexID(nX, nY, nZ + 1) + 1;
	case 5:
		return GetVertexID(nX + 1, nY, nZ + 1);
	case 6:
		return GetVertexID(nX, nY + 1, nZ + 1) + 1;
	case 7:
		return GetVertexID(nX, nY, nZ + 1);
	case 8:
		return GetVertexID(nX, nY, nZ) + 2;
	case 9:
		return GetVertexID(nX + 1, nY, nZ) + 2;
	case 10:
		return GetVertexID(nX + 1, nY + 1, nZ) + 2;
	case 11:
		return GetVertexID(nX, nY + 1, nZ) + 2;
	default:
		// Invalid edge no.
		return -1;
	}
}

// version that calculates flat surface normal for each triangle
__global__ void
generateTriangles2(float3 *pos, float3 *norm, uint *compactedVoxelArray, uint *numVertsScanned, farray volume,
float isoValue, uint activeVoxels, uint maxVerts)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (idx > activeVoxels - 1) {
		idx = activeVoxels - 1;
	}

	int voxel = compactedVoxelArray[idx];
	float3 voxelSize = dparam.cellsize / (NXMC / NX);

	// compute position in 3d grid
	int i, j, k;
	getijk(i, j, k, voxel, NXMC, NYMC, NZMC);

	float3 p;
	p.x = i*voxelSize.x;
	p.y = j*voxelSize.y;
	p.z = k*voxelSize.z;

	float field[8];
	field[0] = volume(i, j, k);
	field[1] = volume(i + 1, j, k);
	field[2] = volume(i + 1, j + 1, k);
	field[3] = volume(i, j + 1, k);
	field[4] = volume(i, j, k + 1);
	field[5] = volume(i + 1, j, k + 1);
	field[6] = volume(i + 1, j + 1, k + 1);
	field[7] = volume(i, j + 1, k + 1);

	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// find the vertices where the surface intersects the cube 

	// use shared memory to avoid using local
	__shared__ float3 vertlist[12 * NTHREADS];

	vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[NTHREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[(NTHREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[(NTHREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
	vertlist[(NTHREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[(NTHREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[(NTHREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[(NTHREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
	vertlist[(NTHREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[(NTHREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[(NTHREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[(NTHREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
	__syncthreads();

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
	for (int idx2 = 0; idx2<numVerts; idx2 += 3) {
		uint index = numVertsScanned[voxel] + idx2;

		float3 *v[3];
		uint edge;
		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2);

		v[0] = &vertlist[(edge*NTHREADS) + threadIdx.x];


		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 1);

		v[1] = &vertlist[(edge*NTHREADS) + threadIdx.x];


		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 2);

		v[2] = &vertlist[(edge*NTHREADS) + threadIdx.x];

		// calculate triangle surface normal
		float3 n = calcNormal(v[0], v[1], v[2]);

		/*if (index < (maxVerts - 3)) */{
			pos[index] = *v[0];
			norm[index] = n;

			pos[index + 1] = *v[1];
			norm[index + 1] = n;

			pos[index + 2] = *v[2];
			norm[index + 2] = n;
		}
	}
}

// version that calculates flat surface normal for each triangle
__global__ void
generateTriangles_indices(float3 *pTriVertex, uint *pTriIndices, uint *compactedVoxelArray, farray volume,
float isoValue, uint activeVoxels, uint maxVerts, uint *MCEdgeIdxMapped, uint *numVertsScanned)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (idx > activeVoxels - 1) {
		idx = activeVoxels - 1;
	}

	int voxel = compactedVoxelArray[idx];
	float3 voxelSize = dparam.cellsize / (NXMC / NX);

	// compute position in 3d grid
	int i, j, k;
	getijk(i, j, k, voxel, NXMC, NYMC, NZMC);

	float3 p;
	p.x = i*voxelSize.x;
	p.y = j*voxelSize.y;
	p.z = k*voxelSize.z;

	float field[8];
	field[0] = volume(i, j, k);
	field[1] = volume(i + 1, j, k);
	field[2] = volume(i + 1, j + 1, k);
	field[3] = volume(i, j + 1, k);
	field[4] = volume(i, j, k + 1);
	field[5] = volume(i + 1, j, k + 1);
	field[6] = volume(i + 1, j + 1, k + 1);
	field[7] = volume(i, j + 1, k + 1);

	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// find the vertices where the surface intersects the cube 

	// use shared memory to avoid using local
	__shared__ float3 vertlist[12 * NTHREADS];

	vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[NTHREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[(NTHREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[(NTHREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
	vertlist[(NTHREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[(NTHREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[(NTHREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[(NTHREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
	vertlist[(NTHREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[(NTHREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[(NTHREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[(NTHREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
	__syncthreads();

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
	uint edge, mappededgeidx;
	for (int idx2 = 0; idx2<numVerts; idx2 += 3) {
		uint index = numVertsScanned[voxel] + idx2;	//vertex index to write back, sort by each triangle.

		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2);
		mappededgeidx = MCEdgeIdxMapped[GetEdgeID(i, j, k, edge)];
		pTriIndices[index] = mappededgeidx;		//notice: indices begin from 0.
		pTriVertex[mappededgeidx] = (vertlist[(edge*NTHREADS) + threadIdx.x]);

		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 1);
		mappededgeidx = MCEdgeIdxMapped[GetEdgeID(i, j, k, edge)];
		pTriIndices[index + 1] = mappededgeidx;		//notice: indices begin from 0.
		pTriVertex[mappededgeidx] = (vertlist[(edge*NTHREADS) + threadIdx.x]);

		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 2);
		mappededgeidx = MCEdgeIdxMapped[GetEdgeID(i, j, k, edge)];
		pTriIndices[index + 2] = mappededgeidx;		//notice: indices begin from 0.
		pTriVertex[mappededgeidx] = (vertlist[(edge*NTHREADS) + threadIdx.x]);
	}
}

__global__ void markActiveEdge_MC(uint *outmark, uint *compactedVoxelArray, farray volume, float isoValue, uint activeVoxels)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (idx > activeVoxels - 1) {
		idx = activeVoxels - 1;
	}

	int voxel = compactedVoxelArray[idx];

	// compute position in 3d grid
	int i, j, k;
	getijk(i, j, k, voxel, NXMC, NYMC, NZMC);

	float field[8];
	field[0] = volume(i, j, k);
	field[1] = volume(i + 1, j, k);
	field[2] = volume(i + 1, j + 1, k);
	field[3] = volume(i, j + 1, k);
	field[4] = volume(i, j, k + 1);
	field[5] = volume(i + 1, j, k + 1);
	field[6] = volume(i + 1, j + 1, k + 1);
	field[7] = volume(i, j + 1, k + 1);

	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
	uint edge;
	for (int idxVert = 0; idxVert<numVerts; idxVert++) {
		//下面可能会重复写，但是应该没问题。注意这个函数执行前需要把outmark置0
		edge = tex1Dfetch(triTex, (cubeindex * 16) + idxVert);
		outmark[GetEdgeID(i, j, k, edge)] = 1;
	}
	//debug
	// 	for( int edge=0; edge<12; edge++ )
	// 		outmark[GetEdgeID(i,j,k,edge)] = 1;

}

//以三角形为核心来计算法线，原子写入到点的法线中。注意：法线不要归一化
__global__ void calnormal_k(float3 *ppos, float3 *pnor, int pnum, uint *indices, int indicesnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < indicesnum / 3)		//face number
	{
		int i1 = indices[idx * 3 + 0];
		int i2 = indices[idx * 3 + 1];
		int i3 = indices[idx * 3 + 2];
		float3 p1 = ppos[i1];
		float3 p2 = ppos[i2];
		float3 p3 = ppos[i3];

		//compute
		float3 nor = cross(p2 - p1, p3 - p1);

		//write back
		atomicAdd(&pnor[i1].x, nor.x);
		atomicAdd(&pnor[i2].x, nor.x);
		atomicAdd(&pnor[i3].x, nor.x);
		atomicAdd(&pnor[i1].y, nor.y);
		atomicAdd(&pnor[i2].y, nor.y);
		atomicAdd(&pnor[i3].y, nor.y);
		atomicAdd(&pnor[i1].z, nor.z);
		atomicAdd(&pnor[i2].z, nor.z);
		atomicAdd(&pnor[i3].z, nor.z);
	}
}

__global__ void normalizeTriangleNor_k(float3 *pnor, int pnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < pnum)		//vertex number
	{
		if (length(pnor[idx])>0)
			pnor[idx] = normalize(pnor[idx]);
	}
}

void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable)
{
	checkCudaErrors(cudaMalloc((void**)d_edgeTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc));

	checkCudaErrors(cudaMalloc((void**)d_triTable, 256 * 16 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, triTex, *d_triTable, channelDesc));

	checkCudaErrors(cudaMalloc((void**)d_numVertsTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc));
}

__global__ void arrayproduct_k(float* out, float* x, float *y, int n)
{
	extern __shared__ float sdata[];
	uint tid = threadIdx.x;
	uint i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	sdata[tid] = (i >= n) ? 0 : (x[i] * y[i]);
	__syncthreads();

	for (int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid<s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		out[blockIdx.x] = sdata[0];
}

//z = Ax: A is a sparse matrix, representing the left hand item of Poisson equation.
__global__ void computeAx(farray ans, charray mark, farray x, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (mark[idx] == TYPEFLUID)		//todo: should add typesolid or not.
		{
			int i, j, k;
			getijk(i, j, k, idx);
			float center = x[idx];
			float sum = -6.0f*center;
			float h2_rev = dparam.cellsize.x*dparam.cellsize.x;
			//notice: x必须在AIR类型的格子里是0，下面的式子才正确
			sum += (mark(i + 1, j, k) == TYPEBOUNDARY) ? center : x(i + 1, j, k);
			sum += (mark(i, j + 1, k) == TYPEBOUNDARY) ? center : x(i, j + 1, k);
			sum += (mark(i, j, k + 1) == TYPEBOUNDARY) ? center : x(i, j, k + 1);
			sum += (mark(i - 1, j, k) == TYPEBOUNDARY) ? center : x(i - 1, j, k);
			sum += (mark(i, j - 1, k) == TYPEBOUNDARY) ? center : x(i, j - 1, k);
			sum += (mark(i, j, k - 1) == TYPEBOUNDARY) ? center : x(i, j, k - 1);
			ans[idx] = sum / h2_rev;
		}
		else
			ans[idx] = 0.0f;
	}
}

//Ans = x + a*y
__global__ void pcg_op(charray A, farray ans, farray x, farray y, float a, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (A[idx] == TYPEFLUID)
			ans[idx] = x[idx] + a*y[idx];
		else
			ans[idx] = 0.0f;
	}
}

__global__ void buildprecondition_pcg(farray P, charray mark, farray ans, farray input, int n)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<n)
	{
		ans[idx] = 1.0f / 6 * input[idx];
	}
}

__global__ void copyParticle2GL_vel_k(float3* ppos, float3 *pvel, float *pmass, char *pflag, int pnum, float *renderpos, float *rendercolor)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		renderpos[idx * 3] = ppos[idx].x;
		renderpos[idx * 3 + 1] = ppos[idx].y;
		renderpos[idx * 3 + 2] = ppos[idx].z;

		if (pflag[idx] == TYPEFLUID)
		{
			rendercolor[idx * 3] = 1.0f;
			rendercolor[idx * 3 + 1] = 0.0f;
			rendercolor[idx * 3 + 2] = 0.0f;
		}
		else if (pflag[idx] == TYPEAIR)
		{
			rendercolor[idx * 3] = 0.0f;
			rendercolor[idx * 3 + 1] = 0.0f;
			rendercolor[idx * 3 + 2] = 1.0f;
		}
		else if (pflag[idx] == TYPESOLID)
		{
			rendercolor[idx * 3] = 0.0f;
			rendercolor[idx * 3 + 1] = 1.0f;
			rendercolor[idx * 3 + 2] = 0.0f;
		}
	}
}

__global__ void copyParticle2GL_radius_k(float3* ppos, float *pmass, char *pflag, int pnum, float *renderpos, float *rendercolor, float minmass)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		renderpos[idx * 3] = ppos[idx].x;
		renderpos[idx * 3 + 1] = ppos[idx].y;
		renderpos[idx * 3 + 2] = ppos[idx].z;

		minmass *= 1.2f;		//trick
		float rate = (pmass[idx] - minmass*dparam.m0) / (dparam.m0 - minmass*dparam.m0);
		rate = fmax(0.0f, fmin(1.0f, rate));
		{
			float3 color = mapColorBlue2Red(powf(rate, 1.0f / 3)*6.0f);
			rendercolor[idx * 3] = color.x;
			rendercolor[idx * 3 + 1] = color.y;
			rendercolor[idx * 3 + 2] = color.z;
		}
	}
}



__device__ inline void atomicaddfloat3(float3 *a, int idx, float3 b)
{
	atomicAdd(&a[idx].x, b.x);
	atomicAdd(&a[idx].y, b.y);
	atomicAdd(&a[idx].z, b.z);
}

__global__ void smooth_computedisplacement(float3 *displacement, int *weight, float3 *ppos, uint *indices, int trianglenum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<trianglenum)
	{
		uint p1 = indices[idx * 3];
		uint p2 = indices[idx * 3 + 1];
		uint p3 = indices[idx * 3 + 2];

		atomicaddfloat3(displacement, p1, ppos[p2] - ppos[p1]);
		atomicaddfloat3(displacement, p1, ppos[p3] - ppos[p1]);
		atomicaddfloat3(displacement, p2, ppos[p1] - ppos[p2]);
		atomicaddfloat3(displacement, p2, ppos[p3] - ppos[p2]);
		atomicaddfloat3(displacement, p3, ppos[p1] - ppos[p3]);
		atomicaddfloat3(displacement, p3, ppos[p2] - ppos[p3]);
		atomicAdd(&weight[p1], 2);
		atomicAdd(&weight[p2], 2);
		atomicAdd(&weight[p3], 2);
	}
}

__global__ void smooth_addDisplacement(float3 *displacement, int *weight, float3 *ppos, int vertexnum, float param)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<vertexnum)
	{
		if (weight[idx]>0)
			ppos[idx] += param * displacement[idx] / weight[idx];
		displacement[idx] = make_float3(0.0f);
		weight[idx] = 0;
	}
}

//diffuse density field.
__global__ void diffuse_dense(farray outp, farray inp, charray mark, float alpha, float beta)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < outp.xn * outp.yn * outp.zn)
	{
		float resp = 0;
		float p1, p2, p3, p4, p5, p6;
		float p0 = inp[idx];
		int i, j, k;
		getijk(i, j, k, idx, outp.xn, outp.yn, outp.zn);
		if (mark(i, j, k) == TYPEBOUNDARY)
			outp[idx] = 0.0f;
		else
		{
			p1 = (mark(i + 1, j, k) == TYPEBOUNDARY) ? p0 : inp(i + 1, j, k);
			p2 = (mark(i, j + 1, k) == TYPEBOUNDARY) ? p0 : inp(i, j + 1, k);
			p3 = (mark(i, j, k + 1) == TYPEBOUNDARY) ? p0 : inp(i, j, k + 1);
			p4 = (mark(i - 1, j, k) == TYPEBOUNDARY) ? p0 : inp(i - 1, j, k);
			p5 = (mark(i, j - 1, k) == TYPEBOUNDARY) ? p0 : inp(i, j - 1, k);
			p6 = (mark(i, j, k - 1) == TYPEBOUNDARY) ? p0 : inp(i, j, k - 1);
			resp = (p1 + p2 + p3 + p4 + p5 + p6 + alpha*p0) / beta;
			outp[idx] = resp;
		}
	}
}

//diffuse velocity field.
__global__ void diffuse_velocity(farray outv, farray inv, float alpha, float beta)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < outv.xn * outv.yn * outv.zn)
	{
		float resp = 0;
		float p1, p2, p3, p4, p5, p6;
		float p0 = inv[idx];
		int i, j, k;
		getijk(i, j, k, idx, outv.xn, outv.yn, outv.zn);
		if (i == 0 || j == 0 || k == 0 || i >= outv.xn - 1 || j >= outv.yn - 1 || k >= outv.zn - 1)
			outv[idx] = p0;
		else
		{
			p1 = inv(i + 1, j, k);
			p2 = inv(i, j + 1, k);
			p3 = inv(i, j, k + 1);
			p4 = inv(i - 1, j, k);
			p5 = inv(i, j - 1, k);
			p6 = inv(i, j, k - 1);
			resp = (p1 + p2 + p3 + p4 + p5 + p6 + alpha*p0) / beta;
			outv[idx] = resp;
		}
	}
}

//maxLength, hashPoints是输出：最长边（每个block里），每个三角形一个用来hash的点
__global__ void createAABB_q(float3* points, int nPoints, uint3* faces, int nFaces, float *maxLength, float3* hashPoints)
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= nFaces)
		return;

	__shared__ float maxArray[256];

	uint p1 = faces[index].x;
	uint p2 = faces[index].y;
	uint p3 = faces[index].z;

	//得到三角形的三个顶点
	float3 px = points[p1];
	float3 py = points[p2];
	float3 pz = points[p3];

	AABB aabb;
	aabb.xMin = (px.x>py.x) ? py.x : px.x;
	aabb.xMin = (aabb.xMin>pz.x) ? pz.x : aabb.xMin;
	aabb.xMax = (px.x<py.x) ? py.x : px.x;
	aabb.xMax = (aabb.xMax<pz.x) ? pz.x : aabb.xMax;

	aabb.yMin = (px.y>py.y) ? py.y : px.y;
	aabb.yMin = (aabb.yMin>pz.y) ? pz.y : aabb.yMin;
	aabb.yMax = (px.y<py.y) ? py.y : px.y;
	aabb.yMax = (aabb.yMax<pz.y) ? pz.y : aabb.yMax;

	aabb.zMin = (px.z>py.z) ? py.z : px.z;
	aabb.zMin = (aabb.zMin>pz.z) ? pz.z : aabb.zMin;
	aabb.zMax = (px.z<py.z) ? py.z : px.z;
	aabb.zMax = (aabb.zMax<pz.z) ? pz.z : aabb.zMax;

	float tempMaxLength = aabb.xMax - aabb.xMin;
	tempMaxLength = (tempMaxLength>aabb.yMax - aabb.yMin) ? (tempMaxLength) : (aabb.yMax - aabb.yMin);
	tempMaxLength = (tempMaxLength>aabb.zMax - aabb.zMin) ? (tempMaxLength) : (aabb.zMax - aabb.zMin);

	maxArray[threadIdx.x] = tempMaxLength;
	hashPoints[index] = make_float3((aabb.xMin + aabb.xMax) / 2, (aabb.yMin + aabb.yMax) / 2, (aabb.zMin + aabb.zMax) / 2);

	__syncthreads();

	for (int i = blockDim.x / 2; i>0; i /= 2)
	{
		if (threadIdx.x < i)
			maxArray[threadIdx.x] = max(maxArray[threadIdx.x], maxArray[i + threadIdx.x]);
		__syncthreads();
	}

	if (threadIdx.x == 0)
		maxLength[blockIdx.x] = maxArray[0];
}


__global__	void calcHash_radix_q(
	uint2*   gridParticleIndex, // output
	float3* posArray,               // input: positions
	uint    numParticles,
	float3 t_min,
	float3 t_max)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	float3 pos = posArray[index];
	uint hash;
	int gz = (pos.z - t_min.z) / dparam.triHashSize.z;
	int gy = (pos.y - t_min.y) / dparam.triHashSize.y;
	int gx = (pos.x - t_min.x) / dparam.triHashSize.x;
	if (gx < 0 || gx > dparam.triHashRes.x - 1 || gy < 0 || gy > dparam.triHashRes.y - 1 || gz < 0 || gz > dparam.triHashRes.z - 1)
		hash = CELL_UNDEF;
	else
		hash = __mul24(__mul24(gz, (int)dparam.triHashRes.y) + gy, (int)dparam.triHashRes.x) + gx;

	// store grid hash and particle index
	gridParticleIndex[index] = make_uint2(hash, index);
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStart_radix_q(uint*   cellStart,        // output: cell start index
uint*   cellEnd,          // output: cell end index
uint3* sortedFaces,
uint2 *  gridParticleHash, // input: sorted grid hashes
uint3* oldFaces,
uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = gridParticleHash[index].x;

		// Load hash data into shared memory so that we can look 
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1].x;
		}
	}

	__syncthreads();

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleHash[index].y;

		sortedFaces[index] = oldFaces[sortedIndex];       // see particles_kernel.cuh
	}
}

__global__ void calculateNormal(float3* points, uint3* faces, float3* normals, int num)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index < num)
	{
		uint3 face = faces[index];

		float3 v1 = points[face.x];
		float3 v2 = points[face.y];
		float3 v3 = points[face.z];

		float3 tmp;
		tmp.x = (v1.y - v2.y)*(v1.z - v3.z) - (v1.z - v2.z)*(v1.y - v3.y);
		tmp.y = (v1.z - v2.z)*(v1.x - v3.x) - (v1.x - v2.x)*(v1.z - v3.z);
		tmp.z = (v1.x - v2.x)*(v1.y - v3.y) - (v1.y - v2.y)*(v1.x - v3.x);

		normals[index] = normalize(tmp);
	}
}

//temp_yanglp: 检测一个小球与三角形是否相交，求出对粒子作用的顶点权重，返回值为负数，表示没有相交，正数表示相交
__device__ float IntersectTriangle_q(float3& pos, float radius, float3& v0, float3& v1, float3& v2, float3 n)
{
	
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash_q(int3 gridPos)
{
	return __umul24(__umul24(gridPos.z, dparam.triHashRes.y), dparam.triHashRes.x) + __umul24(gridPos.y, dparam.triHashRes.x) + gridPos.x;
}

// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
float3  pos,
float radius,
float3* surPoints,
uint3* surIndex,
float3* surfaceNor,
uint*   cellStart,
uint*   cellEnd,
int scene)
{
	uint gridHash = calcGridHash_q(gridPos);

	float dis_n, wib = 0;
	float3 force = make_float3(0.0f);
	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	if (startIndex != CELL_UNDEF) {        // cell is not empty
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];
		for (uint j = startIndex; j<endIndex; j++) {
			//cuPrintf("j=%d\n", j);
			dis_n = IntersectTriangle_q(pos, radius, surPoints[surIndex[j].x], surPoints[surIndex[j].y], surPoints[surIndex[j].z], surfaceNor[j]);
			wib = 1 - dis_n / radius;
			if (dis_n >= 0 && wib > 0.00001)
			{
				force += (radius - dis_n) * (surfaceNor[j]) * 10;
			}
		}
	}

	return force;
}

__device__ void mindis_cell(float& mindisair, float& mindisfluid, float3 gpos, float3 *pos, char *parflag, float *pmass, uint *gridstart, uint  *gridend, int gidx, float radius)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis;
	for (uint p = start; p<end; ++p)
	{
		dis = length(pos[p] - gpos);//减掉半径，后面的数是较正一下  
	//	dis = fabs(length(pos[p] - gpos))- radius;//  依据mantaflow	
		if (parflag[p] == TYPEAIR || parflag[p] == TYPEAIRSOLO)//todo: 是不是加上SOLO的类型以防止ls随着标记变化的突变？
			mindisair = (dis<mindisair) ? dis : mindisair;
		else if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID)
			mindisfluid = (dis<mindisfluid) ? dis : mindisfluid;
	}
}


//MultiFLIP for Energetic Two-Phase Fluid Simulation
__global__ void genlevelset(farray lsfluid, farray lsair, charray mark, float3 *pos, char *parflag, float *pmass, uint *gridstart, uint  *gridend, float fMCDensity, float offset, charray sandmark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)		//每个格子一个值
	{
		//float ls;
		float h = dparam.cellsize.x;
		mark[idx] = TYPEVACUUM;
		sandmark[idx] = TYPEVACUUM;
		float r = 0.5f*h;		//0.36f*h; 
		//float r = 0.5*sqrt(3.)*1.01*2.5;		//修改为0.5*1.01 依据mantaflow
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NX, NY, NZ);

		float3 gpos = (make_float3(i, j, k) + make_float3(0.5f, 0.5f, 0.5f))*dparam.cellsize.x; // shifted by half cell
		float mindisair = 2.5f*h, mindisfluid = 2.5f*h;	//2.5 cellsize
		//float mindisair = r, mindisfluid = r;	//修正 mindis- 为 r 依据mantaflow
		int level = 2;
		for (int di = -level; di <= level; ++di) for (int dj = -level; dj <= level; ++dj) for (int dk = -level; dk <= level; ++dk)	//周围27个格子就行
		{
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				mindis_cell(mindisair, mindisfluid, gpos, pos, parflag, pmass, gridstart, gridend, getidx(i + di, j + dj, k + dk), r);
			}
		}
		mindisair -= r;			//注释掉 依据mataflow
		mindisfluid -= r;

		lsfluid[idx] = mindisfluid;
	//	lsair[idx] = mindisair - offset*h;	//todo: 这里略微向外扩张了一下气体的ls，避免气体粒子correctpos时向内收缩导到气泡体积的减小。注意：这个修正会导致markgrid的不对，因此流体mark会大一层，其流动会受很大影响
		lsair[idx] = mindisair;
	}
}


__device__ void sumcell_fluidSolid(float3 &usum, float &weight, float3 gpos, float3 *pos, float3 *vel, float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID || parflag[p] == TYPESPRAY)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
			w = mass[p] * sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p];
		}
	}
}

//for APIC
//output:usum   weight
__device__ void sumcell_Solid_APIC(matrix3x3 *B, matrix3x3 &tensorD, float3 &usum, float &gmass, float3 gpos, float3 *pos, float3 *vel, float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, weight, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	float3 dis;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPESOLID)
		{
			//previous weight
			//dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);
			//weight = sharp_kernel(dis2, RE);
			//trilinear weight
			dis = gpos - pos[p] * scale; //range [0,1]
			dis.x = (dis.x>0) ? dis.x : -dis.x;
			dis.y = (dis.y>0) ? dis.y : -dis.y;
			dis.z = (dis.x>0) ? dis.z : -dis.z;
			weight = fmaxf((1.0 - dis.x)*(1.0 - dis.y)*(1.0 - dis.z), 0.0f);
			gmass += mass[p] * weight;
			//tensorD += mul(gpos - pos[p]) * sharp_kernel(dis2, RE);
			//usum += w*vel[p];
			//use weight gradients instead of calculating tensorD
			float3 gradient = make_float3(0.0f);
			gradient.x = (1.0 - dis.y)*(1.0 - dis.z);
			gradient.y = (1.0 - dis.x)*(1.0 - dis.z);
			gradient.z = (1.0 - dis.x)*(1.0 - dis.y);
			//usum += weight * mass[p] * vel[p] + B[p] * gradient * mass[p];
			usum += weight * mass[p] * vel[p];
		}
	}
}

__device__ void sumcell_Fluid_APIC(int axis, float3 *cpx, float3 *cpy, float3 *cpz, float &usuma, float &gmass, float3 gpos, float3 *pos, float3 *vel, float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float3 cpa, ea;
	float weight;
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	float3 dis;
	for (uint p = start; p<end; ++p)
	{
		if ((parflag[p] == TYPEFLUID) || (parflag[p] == TYPESPRAY))
		{
			//gpos = (i, j+0.5, k+0.5)
			//previous weight
			//dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);
			//weight = sharp_kernel(dis2, RE);
			//trilinear weight
			dis = gpos - pos[p] * scale; //range [0,1]
			dis.x = (dis.x>0) ? dis.x : -dis.x;
			dis.y = (dis.y>0) ? dis.y : -dis.y;
			dis.z = (dis.x>0) ? dis.z : -dis.z;
			weight = fmaxf((1.0 - dis.x)*(1.0 - dis.y)*(1.0 - dis.z), 0.0f);
			gmass += mass[p] * weight;
			if (axis == 1){
				ea = make_float3(1.0f, 0.0f, 0.0f);
				cpa = cpx[p];
			}
			else if (axis == 2){
				ea = make_float3(0.0f, 1.0f, 0.0f);
				cpa = cpy[p];
			}
			else if (axis == 3){
				ea = make_float3(0.0f, 0.0f, 1.0f);
				cpa = cpz[p];
			}
			usuma += mass[p] * weight * (mul_float(ea, vel[p]) + mul_float(cpa, (gpos / scale - pos[p])));
		}
	}
}

//for APIC

__global__ void mapvelp2g_k_Fluid_APIC(float3 *cpx, float3 *cpy, float3 *cpz, float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float gmass;
	float usuma;
	float3 gpos;
	if (idx<dparam.gvnum.x)
	{
		// ux
		gmass = 0, usuma = 0.0f;
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -1; di <= 0; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_Fluid_APIC(1, cpx, cpy, cpz, usuma, gmass, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usuma = (gmass>0) ? (usuma / gmass) : 0.0f;
		ux(i, j, k) = usuma;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		gmass = 0, usuma = 0.0f;
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 0; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_Fluid_APIC(2, cpx, cpy, cpz, usuma, gmass, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usuma = (gmass>0) ? (usuma / gmass) : 0.0f;
		uy(i, j, k) = usuma;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		gmass = 0, usuma = 0.0f;
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 0; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_Fluid_APIC(3, cpx, cpy, cpz, usuma, gmass, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usuma = (gmass>0) ? (usuma / gmass) : 0.0f;
		uz(i, j, k) = usuma;
	}
}
__global__ void mapvelp2g_k_Solid_APIC(matrix3x3 *B, float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, f3array nodevel, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float gmass;
	matrix3x3 tensorD;
	float3 gpos, usum;
	if (idx<dparam.gnnum)
	{
		gmass = 0, usum = make_float3(0.0f), tensorD = make_matrix3x3(0.0f);
		getijk(i, j, k, idx, NX + 1, NY + 1, NZ + 1);
		gpos.x = i, gpos.y = j, gpos.z = k;
		for (int di = -1; di <= 0; di++) for (int dj = -1; dj <= 0; dj++) for (int dk = -1; dk <= 0; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_Solid_APIC(B, tensorD, usum, gmass, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum = (gmass>0) ? (usum / gmass) : make_float3(0.0f);
		nodevel(i, j, k) = usum;
	}
}

__global__ void mapvelp2g_k_fluidSolid(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float weight;
	float3 gpos, usum;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -1; di <= 0; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_fluidSolid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.x = (weight>0) ? (usum.x / weight) : 0.0f;
		ux(i, j, k) = usum.x;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 0; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_fluidSolid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.y = (weight>0) ? (usum.y / weight) : 0.0f;
		uy(i, j, k) = usum.y;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 0; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_fluidSolid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.z = (weight>0) ? (usum.z / weight) : 0.0f;
		uz(i, j, k) = usum.z;
	}
}

__device__ void sumcell_air(float3 &usum, float &weight, float3 gpos, float3 *pos, float3 *vel,
	float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEAIR)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
			w = mass[p] * sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p];
		}
	}
}

__global__ void mapvelp2g_k_air(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float weight;
	float3 gpos, usum;
	int rangemax = 2, rangemin = 1;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemin; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_air(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.x = (weight>0) ? (usum.x / weight) : 0.0f;
		ux(i, j, k) = usum.x;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemin; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_air(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.y = (weight>0) ? (usum.y / weight) : 0.0f;
		uy(i, j, k) = usum.y;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemin; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_air(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.z = (weight>0) ? (usum.z / weight) : 0.0f;
		uz(i, j, k) = usum.z;
	}
}
__device__ void sumcell_solid(float3 &usum, float &weight, float3 gpos, float3 *pos, float3 *vel,
	float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPESOLID)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
			w = mass[p] * sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p];
		}
	}
}

__global__ void mapvelp2g_k_solid(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float weight;
	float3 gpos, usum;
	int rangemax = 2, rangemin = 1;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemin; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_solid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.x = (weight>0) ? (usum.x / weight) : 0.0f;
		ux(i, j, k) = usum.x;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemin; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_solid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.y = (weight>0) ? (usum.y / weight) : 0.0f;
		uy(i, j, k) = usum.y;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemin; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_solid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.z = (weight>0) ? (usum.z / weight) : 0.0f;
		uz(i, j, k) = usum.z;
	}
}

//计算散度
__global__ void cptdivergence_bubble(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls, farray sf)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		float ux0, ux1, uy0, uy1, uz0, uz1;
		float jx0, jx1, jy0, jy1, jz0, jz1, J;		//surface tension, [2005]Discontinuous Fluids
		float theta;
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			//ux1
			if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) != TYPEAIR)
				ux1 = waterux(i + 1, j, k), jx1 = 0;
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) != TYPEFLUID)
				ux1 = airux(i + 1, j, k), jx1 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * waterux(i + 1, j, k) + (1 - theta) * airux(i + 1, j, k);
				jx1 = theta * sf(i, j, k) + (1 - theta) * sf(i + 1, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * airux(i + 1, j, k) + (1 - theta) * waterux(i + 1, j, k);
				jx1 = theta * sf(i, j, k) + (1 - theta) * sf(i + 1, j, k);
			}

			//ux0
			if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) != TYPEAIR)
				ux0 = waterux(i, j, k), jx0 = 0;
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) != TYPEFLUID)
				ux0 = airux(i, j, k), jx0 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * waterux(i, j, k) + (1 - theta) * airux(i, j, k);
				jx0 = theta*sf(i, j, k) + (1 - theta)*sf(i - 1, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * airux(i, j, k) + (1 - theta) * waterux(i, j, k);
				jx0 = theta*sf(i, j, k) + (1 - theta)*sf(i - 1, j, k);
			}

			//uy1
			if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) != TYPEAIR)
				uy1 = wateruy(i, j + 1, k), jy1 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) != TYPEFLUID)
				uy1 = airuy(i, j + 1, k), jy1 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * wateruy(i, j + 1, k) + (1 - theta) * airuy(i, j + 1, k);
				jy1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j + 1, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * airuy(i, j + 1, k) + (1 - theta) * wateruy(i, j + 1, k);
				jy1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j + 1, k);
			}

			//uy0
			if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) != TYPEAIR)
				uy0 = wateruy(i, j, k), jy0 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) != TYPEFLUID)
				uy0 = airuy(i, j, k), jy0 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * wateruy(i, j, k) + (1 - theta) * airuy(i, j, k);
				jy0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j - 1, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * airuy(i, j, k) + (1 - theta) * wateruy(i, j, k);
				jy0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j - 1, k);
			}

			//uz1
			if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) != TYPEAIR)
				uz1 = wateruz(i, j, k + 1), jz1 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) != TYPEFLUID)
				uz1 = airuz(i, j, k + 1), jz1 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * wateruz(i, j, k + 1) + (1 - theta) * airuz(i, j, k + 1);
				jz1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k + 1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * airuz(i, j, k + 1) + (1 - theta) * wateruz(i, j, k + 1);
				jz1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k + 1);
			}

			//uz0
			if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) != TYPEAIR)
				uz0 = wateruz(i, j, k), jz0 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) != TYPEFLUID)
				uz0 = airuz(i, j, k), jz0 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * wateruz(i, j, k) + (1 - theta) * airuz(i, j, k);
				jz0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k - 1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * airuz(i, j, k) + (1 - theta) * wateruz(i, j, k);
				jz0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k - 1);
			}

			J = (jx1 - jx0 + jy1 - jy0 + jz1 - jz0) / h / h;

			div = (ux1 - ux0 + uy1 - uy0 + uz1 - uz0) / h;
			div += J;	//surfacetension
		}

		outdiv[idx] = div;
	}
}


__global__ void cptdivergence_bubble2(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		float ux0, ux1, uy0, uy1, uz0, uz1;
		float theta;
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			//ux1
			if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) != TYPEAIR)
				ux1 = waterux(i + 1, j, k);
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) != TYPEFLUID)
				ux1 = airux(i + 1, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * waterux(i + 1, j, k) + (1 - theta) * airux(i + 1, j, k);
				//ux1 = airux(i+1,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * airux(i + 1, j, k) + (1 - theta) * waterux(i + 1, j, k);
				//ux1 = airux(i+1,j,k);
			}

			//ux0
			if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) != TYPEAIR)
				ux0 = waterux(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) != TYPEFLUID)
				ux0 = airux(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * waterux(i, j, k) + (1 - theta) * airux(i, j, k);
				//ux0 = airux(i,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * airux(i, j, k) + (1 - theta) * waterux(i, j, k);
				//ux0 = airux(i,j,k);
			}

			//uy1
			if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) != TYPEAIR)
				uy1 = wateruy(i, j + 1, k);
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) != TYPEFLUID)
				uy1 = airuy(i, j + 1, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * wateruy(i, j + 1, k) + (1 - theta) * airuy(i, j + 1, k);
				//uy1 = airuy(i,j+1,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * airuy(i, j + 1, k) + (1 - theta) * wateruy(i, j + 1, k);
				//uy1 = airuy(i,j+1,k);
			}

			//uy0
			if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) != TYPEAIR)
				uy0 = wateruy(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) != TYPEFLUID)
				uy0 = airuy(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * wateruy(i, j, k) + (1 - theta) * airuy(i, j, k);
				//	uy0 = airuy(i,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * airuy(i, j, k) + (1 - theta) * wateruy(i, j, k);
				//uy0 = airuy(i,j,k);
			}

			//uz1
			if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) != TYPEAIR)
				uz1 = wateruz(i, j, k + 1);
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) != TYPEFLUID)
				uz1 = airuz(i, j, k + 1);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * wateruz(i, j, k + 1) + (1 - theta) * airuz(i, j, k + 1);
				//uz1 = airuz(i,j,k+1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * airuz(i, j, k + 1) + (1 - theta) * wateruz(i, j, k + 1);
				//uz1 = airuz(i,j,k+1);
			}

			//uz0
			if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) != TYPEAIR)
				uz0 = wateruz(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) != TYPEFLUID)
				uz0 = airuz(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * wateruz(i, j, k) + (1 - theta) * airuz(i, j, k);
				//uz0 = airuz(i,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * airuz(i, j, k) + (1 - theta) * wateruz(i, j, k);
				//uz0 = airuz(i,j,k);
			}
			div = (ux1 - ux0 + uy1 - uy0 + uz1 - uz0) / h;
		}

		outdiv[idx] = div;
	}
}

__global__ void cptdivergence_bubble3(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		float ux0, ux1, uy0, uy1, uz0, uz1;
		float theta;
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			//ux1
			if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) != TYPEAIR)
				ux1 = waterux(i + 1, j, k);
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) != TYPEFLUID)
				ux1 = airux(i + 1, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i+1,j,k)-ls(i,j,k));
				// 				ux1 = theta * waterux(i+1,j,k) + (1-theta) * airux(i+1,j,k);
				ux1 = airux(i + 1, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i+1,j,k)-ls(i,j,k));
				// 				ux1 = theta * airux(i+1,j,k) + (1-theta) * waterux(i+1,j,k);
				ux1 = airux(i + 1, j, k);
			}

			//ux0
			if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) != TYPEAIR)
				ux0 = waterux(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) != TYPEFLUID)
				ux0 = airux(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i-1,j,k)-ls(i,j,k));
				// 				ux0 = theta * waterux(i,j,k) + (1-theta) * airux(i,j,k);
				ux0 = airux(i, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i-1,j,k)-ls(i,j,k));
				// 				ux0 = theta * airux(i,j,k) + (1-theta) * waterux(i,j,k);
				ux0 = airux(i, j, k);
			}

			//uy1
			if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) != TYPEAIR)
				uy1 = wateruy(i, j + 1, k);
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) != TYPEFLUID)
				uy1 = airuy(i, j + 1, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j+1,k)-ls(i,j,k));
				// 				uy1 = theta * wateruy(i,j+1,k) + (1-theta) * airuy(i,j+1,k);
				uy1 = airuy(i, j + 1, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j+1,k)-ls(i,j,k));
				// 				uy1 = theta * airuy(i,j+1,k) + (1-theta) * wateruy(i,j+1,k);
				uy1 = airuy(i, j + 1, k);
			}

			//uy0
			if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) != TYPEAIR)
				uy0 = wateruy(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) != TYPEFLUID)
				uy0 = airuy(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j-1,k)-ls(i,j,k));
				// 				uy0 = theta * wateruy(i,j,k) + (1-theta) * airuy(i,j,k);
				uy0 = airuy(i, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j-1,k)-ls(i,j,k));
				// 				uy0 = theta * airuy(i,j,k) + (1-theta) * wateruy(i,j,k);
				uy0 = airuy(i, j, k);
			}

			//uz1
			if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) != TYPEAIR)
				uz1 = wateruz(i, j, k + 1);
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) != TYPEFLUID)
				uz1 = airuz(i, j, k + 1);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j,k+1)-ls(i,j,k));
				// 				uz1 = theta * wateruz(i,j,k+1) + (1-theta) * airuz(i,j,k+1);
				uz1 = airuz(i, j, k + 1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j,k+1)-ls(i,j,k));
				// 				uz1 = theta * airuz(i,j,k+1) + (1-theta) * wateruz(i,j,k+1);
				uz1 = airuz(i, j, k + 1);
			}

			//uz0
			if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) != TYPEAIR)
				uz0 = wateruz(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) != TYPEFLUID)
				uz0 = airuz(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR)
			{
				// 				theta=(0.0f-ls(i,j,k))/(ls(i,j,k-1)-ls(i,j,k));
				// 				uz0 = theta * wateruz(i,j,k) + (1-theta) * airuz(i,j,k);
				uz0 = airuz(i, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID)
			{
				// 				theta=(0.0f-ls(i,j,k))/(ls(i,j,k-1)-ls(i,j,k));
				// 				uz0 = theta * airuz(i,j,k) + (1-theta) * wateruz(i,j,k);
				uz0 = airuz(i, j, k);
			}

			div = (ux1 - ux0 + uy1 - uy0 + uz1 - uz0) / h;
		}

		outdiv[idx] = div;
	}
}


__global__ void subGradPress_bubble(farray p, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float h = dparam.cellsize.x;
	if (idx<dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>0 && i<NX)		//look out for this condition
			ux(i, j, k) -= (p(i, j, k) - p(i - 1, j, k)) / h;
	}
	if (idx<dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if (j>0 && j<NY)		//look out for this condition
			uy(i, j, k) -= (p(i, j, k) - p(i, j - 1, k)) / h;
	}
	if (idx<dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if (k>0 && k<NZ)		//look out for this condition
			uz(i, j, k) -= (p(i, j, k) - p(i, j, k - 1)) / h;
	}
}

//calculate strain rate tensor matrix3x3 D and apply yield condition
__global__ void computeD(matarray D, matarray sigma, farray press, charray sandmark, farray ux, farray uy, farray uz, charray mmark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		
		int i, j, k;
		getijk(i, j, k, idx);
		
		if (mmark(i, j, k) != TYPEFLUID)
			return;
		else
		{
			//????? uy_y = uy.. not ux!!
			float ux_x = (ux(i + 1, j, k) - ux(i, j, k)) / dparam.cellsize.x;
			float uy_y = (uy(i, j + 1, k) - uy(i, j, k)) / dparam.cellsize.y;
			float uz_z = (uz(i, j, k + 1) - uz(i, j, k)) / dparam.cellsize.z;
			float ux_y, ux_z, uy_x, uy_z, uz_x, uz_y;
			//uy_x & uz_x
			if (i == 0 || i == NX - 1)
			{
				uy_x = 0;
				uz_x = 0;
			}
			else 
			{ 
				uy_x = (uy(i + 1, j, k) + uy(i + 1, j + 1, k) - uy(i - 1, j, k) - uy(i - 1, j + 1, k)) / (4.0f*dparam.cellsize.x);
				uz_x = (uz(i + 1, j, k) + uz(i + 1, j, k + 1) - uz(i - 1, j, k) - uz(i - 1, j, k + 1)) / (4.0f*dparam.cellsize.x);
			}

			//ux_y & uz_y
			if (j == 0 || j == NY - 1)
			{
				ux_y = 0;
				uz_y = 0;
			}
			else
			{
				ux_y = (ux(i, j + 1, k) + ux(i + 1, j + 1, k) - ux(i, j - 1, k) - ux(i + 1, j - 1, k)) / (4.0f*dparam.cellsize.y);
				uz_y = (uz(i, j + 1, k) + uz(i, j + 1, k + 1) - uz(i, j - 1, k) - uz(i, j - 1, k + 1)) / (4.0f*dparam.cellsize.y);
			}

			//ux_z & uy_z
			if (k == 0 || k == NZ - 1)
			{
				ux_z = 0;
				uy_z = 0;
			}
			else
			{
				ux_z = (ux(i, j, k + 1) + ux(i + 1, j, k + 1) - ux(i, j, k - 1) - ux(i + 1, j, k - 1)) / (4.0f*dparam.cellsize.z);
				uy_z = (uy(i, j, k + 1) + uy(i, j + 1, k + 1) - uy(i, j, k - 1) - uy(i, j + 1, k - 1)) / (4.0f*dparam.cellsize.z);
			}

			D(i, j, k).x00 = ux_x;		D(i, j, k).x01 = 0.5f * (ux_y + uy_x);		D(i, j, k).x02 = 0.5f * (ux_z + uz_x);
			D(i, j, k).x10 = 0.5f * (uy_x + ux_y);		D(i, j, k).x11 = uy_y;		D(i, j, k).x12 = 0.5f * (uy_z + uz_y);
			D(i, j, k).x20 = 0.5f * (ux_z + uz_x);		D(i, j, k).x21 = 0.5f * (uy_z + uz_y);		D(i, j, k).x22 = uz_z;

			//d = |D|
			float d = sqrtf(D(i, j, k).x00 * D(i, j, k).x00 + D(i, j, k).x01 * D(i, j, k).x01 + D(i, j, k).x02 * D(i, j, k).x02 +
				D(i, j, k).x10 * D(i, j, k).x10 + D(i, j, k).x11 * D(i, j, k).x11 + D(i, j, k).x12 * D(i, j, k).x12 +
				D(i, j, k).x20 * D(i, j, k).x20 + D(i, j, k).x21 * D(i, j, k).x21 + D(i, j, k).x22 * D(i, j, k).x22);

			//!!!!!!press here equals [dt * press / rho]
			matrix3x3 sigma_f = D(i, j, k) * (-dparam.sin_phi * press(i, j, k) * sqrtf(3.f) / d);
			matrix3x3 sigma_rigid = D(i, j, k) * (-dparam.cellsize.x * dparam.cellsize.x);
			float s_rigid = d * dparam.cellsize.x * dparam.cellsize.x;
			float s_f = dparam.sin_phi * press(i, j, k) * sqrtf(3.f);

			//DEBUG Print
			//printf("press = %f|| s_rigid = %f, s_f = %f\n", press(i, j, k), s_rigid, abs(s_f));
			if (s_rigid <= abs(s_f) + dparam.cohesion)
			//if (s_f <= s_rigid + dparam.cohesion)
			{
				sandmark(i, j, k) = TYPERIGID;
				sigma(i, j, k) = sigma_rigid;
			}
			else
			{
				sandmark(i, j, k) = TYPEYIELD;
				sigma(i, j, k) = sigma_f ;
			}
		}
	}
}

__global__ void CalculateRigidcell(farray ux, farray uy, farray uz, charray sandmark, int* rigidcellnum, float3* v_bar, float3* w_bar, float3* x_bar)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float h = dparam.cellsize.x;
		if (sandmark(i, j, k) == TYPESPRAY)
			return;

		if (sandmark(i, j, k) == TYPERIGID)
		{
			//consider mass of every cell is the same
			float3 vel = make_float3(0.0f);
			//float3 vi = make_float3((ux(i, j, k) + ux(i + 1, j, k)) / 2.0f, (uy(i, j, k) + uy(i, j + 1, k)) / 2.0f, (uz(i, j, k) + uz(i, j, k + 1)) / 2.0f);
			float3 vi = make_float3(ux(i, j, k), uy(i, j, k), uz(i, j, k));
			float3 ri = make_float3((i*h + 0.5f*h), (j*h + 0.5f*h), (k*h + 0.5f*h));
			float3 xi = ri;
			
			float3 ri_vi = make_float3((ri.y*vi.z - ri.z*vi.y), (ri.z*vi.x - ri.x*vi.z), (ri.x*vi.y - ri.y*vi.x));

			atomicAdd(&rigidcellnum[0], 1);
			atomicaddfloat3(v_bar, 0, vi);
			atomicaddfloat3(w_bar, 0, ri_vi);
			atomicaddfloat3(x_bar, 0, xi);
		}
	}
}

__global__ void update_cellvelocity(matarray sigma, farray press, f3array yieldvel, charray sandmark, farray ux, farray uy, farray uz, float3* v_bar, float3* w_bar, float3* x_bar, int mframe, int* connect_id)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float h = dparam.cellsize.x;
		//for rigid cells
		if (sandmark(i, j, k) == TYPERIGID)
		{
			int group_id = connect_id[idx];

			//if (0)
			{
				/*float3 x = make_flaot3((i + 0.5) * h - x_bar.x, (j + 0.5) * h - x_bar.y, (k + 0.5) * h - x_bar.z);
				ux(i,j,k)*/
				ux(i, j, k) = v_bar[group_id].x;
				ux(i + 1, j, k) = v_bar[group_id].x;
				uy(i, j, k) = v_bar[group_id].y;
				uy(i + 1, j, k) = v_bar[group_id].y;
				uz(i, j, k) = v_bar[group_id].z;
				uz(i + 1, j, k) = v_bar[group_id].z;
				
			}
			//ux(i, j, k) = 0;
			//uy(i, j, k) = 0;
			//uz(i, j, k) = 0;

		}
		//for yielding cells
		else if (sandmark(i, j, k) == TYPEYIELD)
		{
			float nabla = 0;

			if ((i > 0) && (sandmark(i - 1, j, k) == TYPEYIELD))
			{
				//ux
				nabla = (sigma(i, j, k).x00 - sigma(i - 1, j, k).x00) / h;//x00_x
				nabla += ((sigma(i, j + 1, k).x01 + sigma(i - 1, j + 1, k).x01) - (sigma(i, j - 1, k).x01 + sigma(i - 1, j - 1, k).x01)) / (4 * h);//x01_y
				nabla += ((sigma(i, j, k + 1).x02 + sigma(i - 1, j, k + 1).x02) - (sigma(i, j, k - 1).x02 + sigma(i - 1, j, k - 1).x02)) / (4 * h);//x02_z

				ux(i, j, k) -= nabla;
			}
			if ((j > 0) && (sandmark(i, j - 1, k) == TYPEYIELD))
			{
				//uy
				nabla = (sigma(i, j, k).x11 - sigma(i, j - 1, k).x11) / h;//x11_y
				nabla += ((sigma(i + 1, j, k).x10 + sigma(i + 1, j - 1, k).x10) - (sigma(i - 1, j, k).x10 + sigma(i - 1, j - 1, k).x10)) / (4 * h);//x10_x
				nabla += ((sigma(i, j, k + 1).x12 + sigma(i, j - 1, k + 1).x12) - (sigma(i, j, k - 1).x12 + sigma(i, j - 1, k - 1).x12)) / (4 * h);//x12_z

				uy(i, j, k) -= nabla;
			}
			if ((k > 0) && (sandmark(i, j, k - 1) == TYPEYIELD))
			{
				//uz
				nabla = (sigma(i, j, k).x22 - sigma(i, j, k - 1).x22) / h;//x22_z
				nabla += ((sigma(i, j + 1, k).x21 + sigma(i, j + 1, k - 1).x21) - (sigma(i, j - 1, k).x21 + sigma(i, j - 1, k - 1).x21)) / (4 * h);//x21_y
				nabla += ((sigma(i + 1, j, k).x20 + sigma(i + 1, j, k - 1).x20) - (sigma(i - 1, j, k).x20 + sigma(i - 1, j, k - 1).x20)) / (4 * h);//x20_x

				uz(i, j, k) -= nabla;

			}
		}
		
	}
}

__global__ void frictional_boundary_conditions(farray ux, farray uy, farray uz, int num, float3 wind)
{ 
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	
	if (idx < num)
	{
		int i, j, k;
		getijk(i, j, k, idx);

		float coefficient = 0.f;
		float vel_normal = dparam.dt * (dparam.gravity.z + wind.z);
		
		float3 cellvel = make_float3(0.5f*(ux(i, j, k) + ux(i + 1, j, k)), 0.5f*(uy(i, j, k) + uy(i, j + 1, k)), 0.5f*(uz(i, j, k) + uz(i, j, k + 1)));

		float vel_tangential = sqrtf(ux(i, j, k) * ux(i, j, k) + uy(i, j, k) * uy(i, j, k));
		//float vel_tangential = sqrtf(cellvel.x * cellvel.x + cellvel.y * cellvel.y);
		float vel_xz = sqrtf(ux(i, j, k) * ux(i, j, k) + uz(i, j, k) * uz(i, j, k));
		float vel_yz = sqrtf(uz(i, j, k) * uz(i, j, k) + uy(i, j, k) * uy(i, j, k));
		float delta_x = 0, delta_y = 0, delta_z = 0;

		//if (k<=2)
		//{
		//	ux(i, j, k) = 0;
		//	uy(i, j, k) = 0;
		//	uz(i, j, k) = 0;
		//	return;
		//}

		if (k <= 2 && uz(i, j, k) <= 0)
		{
			if (vel_normal>0)
					return;
			if ((uz(i, j, k) == 0) || (vel_tangential / (-uz(i, j, k)) <= dparam.miu))
			{
				coefficient = 0.f;
			}
			else if (vel_tangential / (-uz(i, j, k)) > dparam.miu)
				coefficient = 1.0f + dparam.miu * uz(i, j, k) / vel_tangential;

			ux(i, j, k) = coefficient * ux(i, j, k);
			uy(i, j, k) = coefficient * uy(i, j, k);
		}

		if (i <= 2 && ux(i, j, k) <= 0)
		{
			if ((ux(i, j, k) == 0) || (vel_yz / (-ux(i, j, k)) <= dparam.miu))
				coefficient = 0.f;
			else if (vel_yz / (-ux(i, j, k)) > dparam.miu)
				coefficient = 1.0f + dparam.miu * ux(i, j, k) / vel_yz;

			uy(i, j, k) = coefficient * uy(i, j, k);
			if (uz(i, j, k) > 0)
				uz(i, j, k) = coefficient * uz(i, j, k);
		}
		else if (i >= NX - 3 && ux(i, j, k)>=0)
		{
			if ((ux(i, j, k) == 0) || (vel_yz / ux(i, j, k) <= dparam.miu))
				coefficient = 0.f;
			else if (vel_yz / ux(i, j, k) > dparam.miu)
				coefficient = 1.0f - dparam.miu * ux(i, j, k) / vel_yz;

			uy(i, j, k) = coefficient * uy(i, j, k);
			if (uz(i, j, k) > 0)
				uz(i, j, k) = coefficient * uz(i, j, k);
		}

		if (j <= 2 && uy(i, j, k) <= 0)
		{
			if ((uy(i, j, k) == 0) || (vel_xz / (-uy(i, j, k)) <= dparam.miu))
				coefficient = 0.f;
			else if (vel_xz / (-uy(i, j, k)) > dparam.miu)
				coefficient = 1.0f + dparam.miu * uy(i, j, k) / vel_xz;

			ux(i, j, k) = coefficient * ux(i, j, k);
			if (uz(i, j, k) > 0)
				uz(i, j, k) = coefficient * uz(i, j, k);
		}
		else if (j >= NY - 3 && uy(i, j, k)>=0)
		{
			if ((uy(i, j, k) == 0) || (vel_xz / uy(i, j, k) <= dparam.miu))
				coefficient = 0.f;
			else if (vel_xz / uy(i, j, k) > dparam.miu)
				coefficient = 1.0f - dparam.miu * uy(i, j, k) / vel_xz;

			ux(i, j, k) = coefficient * ux(i, j, k);
			if (uz(i, j, k) > 0)
				uz(i, j, k) = coefficient * uz(i, j, k);
		}

		
	}
}

//z = Ax: A is a sparse matrix, representing the left hand item of Poisson equation.
__global__ void computeAx_bubble(farray ans, charray mark, farray x, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			int i, j, k;
			getijk(i, j, k, idx);
			float center = x[idx];
			float sum = -6.0f*center;
			float h2_rev = dparam.cellsize.x*dparam.cellsize.x;

			sum += (mark(i + 1, j, k) == TYPEBOUNDARY) ? center : x(i + 1, j, k);
			sum += (mark(i, j + 1, k) == TYPEBOUNDARY) ? center : x(i, j + 1, k);
			sum += (mark(i, j, k + 1) == TYPEBOUNDARY) ? center : x(i, j, k + 1);
			sum += (mark(i - 1, j, k) == TYPEBOUNDARY) ? center : x(i - 1, j, k);
			sum += (mark(i, j - 1, k) == TYPEBOUNDARY) ? center : x(i, j - 1, k);
			sum += (mark(i, j, k - 1) == TYPEBOUNDARY) ? center : x(i, j, k - 1);
			ans[idx] = sum / h2_rev;
		}
		else
			ans[idx] = 0.0f;
	}
}

//Ans = x + a*y
__global__ void pcg_op_bubble(charray A, farray ans, farray x, farray y, float a, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (A[idx] == TYPEFLUID || A[idx] == TYPEAIR)
			ans[idx] = x[idx] + a*y[idx];
		else
			ans[idx] = 0.0f;
	}
}


__global__ void advectparticle_RK2_bubble(float3 *ppos, float3 *pvel, int pnum, f3array solidvel, farray waterux, farray wateruy, farray wateruz,
	farray airux, farray airuy, farray airuz, float dt, char *parflag, VELOCITYMODEL velmode, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if ((parflag[idx] != TYPEFLUID)&&(parflag[idx] != TYPESPRAY))	//对于小的气体粒子AIRSOLO，什么也不更新，跳过
			return;

		//read in
		float3 ipos = ppos[idx], ivel = pvel[idx];
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.cellsize.x));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.cellsize.x));
		char partype = parflag[idx];

		//pos-->grid xyz
		float3 gvel = make_float3(0.0f);
		if (partype == TYPESOLID)
			gvel = getParticleVelFromGrid_APIC(ipos, solidvel);
		else if ((partype == TYPEFLUID) || (partype == TYPESPRAY))
			gvel = getParticleVelFromGrid(ipos, waterux, wateruy, wateruz);
		else if (partype == TYPEAIR)
			gvel = getParticleVelFromGrid(ipos, airux, airuy, airuz);
		else		//TYPEAIRSOLO 有自己的仿真方法，不参与这些仿真
			return;

		if (velmode == CIP || velmode == APIC) /*|| partype==TYPEAIR*/		//todo: 气体粒子用cip模式，减少乱跑的可能
			ivel = gvel;
		else
			ivel = (1 - FLIP_ALPHA)*gvel + FLIP_ALPHA*pvel[idx];

		//mid point: x(n+1/2) = x(n) + 0.5*dt*u(xn)
		float3 midpoint = ipos + gvel * dt * 0.5;
		float3 gvelmidpoint;
		if (partype == TYPESOLID)
			gvelmidpoint = getParticleVelFromGrid_APIC(midpoint, solidvel);
		else if ((partype == TYPEFLUID) || (partype == TYPESPRAY))
			gvelmidpoint = getParticleVelFromGrid(midpoint, waterux, wateruy, wateruz);
		else
			gvelmidpoint = getParticleVelFromGrid(midpoint, airux, airuy, airuz);

		// x(n+1) = x(n) + dt*u(x+1/2)
		ipos += gvelmidpoint * dt;

		

		//check boundary
		float vel_normal = dparam.dt * (dparam.gravity.z + wind.z);
		float vel_xy = sqrtf(ivel.x * ivel.x + ivel.y * ivel.y);
		float vel_xz = sqrtf(ivel.x * ivel.x + ivel.z * ivel.z);
		float vel_yz = sqrtf(ivel.y * ivel.y + ivel.z * ivel.z);
		float coefficient = 1.0f;
		if (ipos.x <= tmin.x)
			ipos.x = tmin.x, ivel.x = 0.0f, coefficient = dparam.miu;
		if (ipos.y <= tmin.y)
			ipos.y = tmin.y, ivel.y = 0.0f, coefficient = dparam.miu;
		if (ipos.z <= tmin.z)
		{
			ipos.z = tmin.z, ivel.z = 0.0f;
			if (vel_xy <= 1.0f)
			{
				ivel.x = 0; 
				ivel.y = 0;
			}
			//if (vel_normal < 0)
			//{
			//	coefficient = fmaxf(0.0f, 1 + (dparam.miu * vel_normal) / sqrtf(ivel.x * ivel.x + ivel.y * ivel.y));
			//	printf("coefficient = %f, vel_normal = %f, vel_tangiential = %f\n", coefficient, ivel.z, sqrtf(ivel.x * ivel.x + ivel.y * ivel.y));
			//}
			
		}

		if (ipos.x >= tmax.x)
			ipos.x = tmax.x, ivel.x = 0.0f, coefficient = dparam.miu;
		if (ipos.y >= tmax.y)
			ipos.y = tmax.y, ivel.y = 0.0f, coefficient = dparam.miu;
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//write back: TYPEAIR+TYPESOLID只更新速度，TYPESOLO之前已经return，TYPEFLUID更新位置和速度。
		pvel[idx] = ivel;// *coefficient;
		//if (pvel[idx].z < 0 && coefficient!=1)
		//{
		//	pvel[idx].z /= coefficient;
		//	//printf("coefficient = %f\n", coefficient);
		//}
		ppos[idx] = ipos;
	}
}

__global__ void mapvelg2p_flip_bubble(float3 *ppos, float3 *vel, char* parflag, int pnum, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		float3 gvel = make_float3(0.0f);
		if (parflag[idx] == TYPEFLUID || parflag[idx] == TYPESOLID || parflag[idx] == TYPESPRAY)
			gvel = getParticleVelFromGrid(ipos, waterux, wateruy, wateruz);
		else if (parflag[idx] == TYPEAIR)
			gvel = getParticleVelFromGrid(ipos, airux, airuy, airuz);

		vel[idx] += gvel;
	}
}


__global__ void compsurfacetension_k(farray sf, charray mark, farray phigrax, farray phigray, farray phigraz, float sigma)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] != TYPEBOUNDARY)
		{
			int i, j, k;
			getijk(i, j, k, idx);

			float len, h = dparam.cellsize.x;
			float res, grax1, gray1, graz1, grax0, gray0, graz0;
			float3 phigracenter = make_float3(phigrax[idx], phigray[idx], phigraz[idx]);
			len = length(phigracenter);
			if (len == 0)
				res = 0;
			else
			{
				phigracenter /= len;

				if (verifycellidx(i + 1, j, k))
				{
					len = length(make_float3(phigrax(i + 1, j, k), phigray(i + 1, j, k), phigraz(i + 1, j, k)));
					if (len == 0)
						grax1 = phigracenter.x;
					else
						grax1 = phigrax(i + 1, j, k) / len;
				}
				else
					grax1 = phigracenter.x;

				if (verifycellidx(i - 1, j, k))
				{
					len = length(make_float3(phigrax(i - 1, j, k), phigray(i - 1, j, k), phigraz(i - 1, j, k)));
					if (len == 0)
						grax0 = phigracenter.x;
					else
						grax0 = phigrax(i - 1, j, k) / len;
				}
				else
					grax0 = phigracenter.x;

				if (verifycellidx(i, j + 1, k))
				{
					len = length(make_float3(phigrax(i, j + 1, k), phigray(i, j + 1, k), phigraz(i, j + 1, k)));
					if (len == 0)
						gray1 = phigracenter.y;
					else
						gray1 = phigray(i, j + 1, k) / len;
				}
				else
					gray1 = phigracenter.y;

				if (verifycellidx(i, j - 1, k))
				{
					len = length(make_float3(phigrax(i, j - 1, k), phigray(i, j - 1, k), phigraz(i, j - 1, k)));
					if (len == 0)
						gray0 = phigracenter.y;
					else
						gray0 = phigray(i, j - 1, k) / len;
				}
				else
					gray0 = phigracenter.y;

				if (verifycellidx(i, j, k + 1))
				{
					len = length(make_float3(phigrax(i, j, k + 1), phigray(i, j, k + 1), phigraz(i, j, k + 1)));
					if (len == 0)
						graz1 = phigracenter.z;
					else
						graz1 = phigraz(i, j, k + 1) / len;
				}
				else
					graz1 = phigracenter.z;
				if (verifycellidx(i, j, k - 1))
				{
					len = length(make_float3(phigrax(i, j, k - 1), phigray(i, j, k - 1), phigraz(i, j, k - 1)));
					if (len == 0)
						graz0 = phigracenter.z;
					else
						graz0 = phigraz(i, j, k - 1) / len;
				}
				else
					graz0 = phigracenter.z;

				res = (grax1 - grax0 + gray1 - gray0 + graz1 - graz0) / h * 0.5f;
				//res = (grax1-phigracenter.x + gray1-phigracenter.y + graz1-phigracenter.z) / h ;
			}

			sf[idx] = res*sigma;
		}
		else
			sf[idx] = 0;
	}
}

__global__ void enforcesurfacetension_p(float3* ppos, float3 *pvel, char *pflag, int pnum, farray lsmerge, farray sf, farray phigrax, farray phigray, farray phigraz, charray mark, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPESOLID/* || pflag[idx]==TYPEAIRSOLO*/ || pflag[idx] == TYPEFLUID)
			return;
		if( (scene != SCENE_MELTANDBOIL&&scene != SCENE_MELTANDBOIL_HIGHRES && pflag[idx] == TYPEAIRSOLO) || ((scene != SCENE_ALL && pflag[idx] == TYPEAIRSOLO)))
			return;

		//1. compute the cell, and get the ls, get sf.
		float3 ipos = ppos[idx];
		float ilsmerge = getScaleFromFrid(ipos, lsmerge);
		float isf = getScaleFromFrid(ipos, sf);
		float3 dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
		float lendir = length(dir);
		if (lendir == 0)
			return;
		float3 f;

		dir /= lendir;
		ilsmerge /= lendir;

		//周围最少一个格子是空气的
		int i, j, k;
		getijkfrompos(i, j, k, ipos);
		int cnt = (mark(i, j, k) == TYPEAIR) ? 1 : 0;
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		if (verifycellidx(i + di, j + dj, k + dk))
		if (mark(i + di, j + dj, k + dk) == TYPEAIR)
			cnt++;
		if (cnt == 0)
			return;

		// if(abs(ls_p)<threshold), enforce a surface tension force, change the velocity.
		if (abs(ilsmerge)<dparam.cellsize.x)
		{
			f = -isf*dir;
			pvel[idx] += f*dparam.dt;
		}
	}
}

//标记levelset里比较大的正数，他们是邻近域内没有粒子的
__global__ void markLS_bigpositive(farray ls, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = ls[idx] / dparam.cellsize.x;
		if (ls[idx] >1.99f)
		{
			ls[idx] = 5.0f;
			mark[idx] = TYPEAIR;	//标记为需要sweep的单元，并非真正的标记 
		}
		else
			mark[idx] = TYPEFLUID;
	}
}

__global__ void setLSback_bigpositive(farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = ls[idx] * dparam.cellsize.x;
	}
}

__global__ void preparels(farray ls, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = -ls[idx] / dparam.cellsize.x;
		if (ls[idx] >0)
		{
			ls[idx] = 5.0f;
			mark[idx] = TYPEAIR;	//标记为需要sweep的单元，并非真正的标记 
		}
		else
			mark[idx] = TYPEFLUID;
	}
}

__global__ void setLSback(farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = -ls[idx] * dparam.cellsize.x;
	}
}

__global__ void mergeLSAndMarkGrid(farray lsmerge, charray mark, farray lsfluid, farray lsair)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx< dparam.gnum)
	{
		float h = dparam.cellsize.x;

		if (lsair[idx] >4.99f * h)
		{
			lsmerge[idx] = lsfluid[idx];
			if (lsfluid[idx]>0)
				mark[idx] = TYPEVACUUM;
			else
				mark[idx] = TYPEFLUID;
		}
		else if (lsfluid[idx]>4.99f*h)
		{
			lsmerge[idx] = lsair[idx];
			if (lsair[idx]>0)
				mark[idx] = TYPEVACUUM;
			else
				mark[idx] = TYPEAIR;
		}
		else if (lsair[idx]>0.8f*h && lsfluid[idx]>0.8f*h)
		{
			mark[idx] = TYPEVACUUM;
			lsmerge[idx] = min(lsfluid[idx], lsair[idx]);
		}
		else
		{
			lsmerge[idx] = (lsfluid[idx] - lsair[idx])*0.5f;
			if (lsmerge[idx]>0)
				mark[idx] = TYPEAIR;
			else
				mark[idx] = TYPEFLUID;
		}
		//todo: 对于气体将出到水面的时候，ls还是会有问题
		int i, j, k;
		getijk(i, j, k, idx);
		if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
			mark[idx] = TYPEBOUNDARY, lsmerge[idx] = -0.5f*h;
		//todo: debug: 
		//lsmerge[idx] = -lsmerge[idx];
	}
}

__global__ void sweepu_k_bubble(farray outux, farray outuy, farray outuz, farray ux, farray uy, farray uz, farray ls, charray mark, char sweepflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	float wx, wy, wz, wsum;		//三个方向上的权重
	if (idx < dparam.gvnum.x)
	{
		//copy
		outux[idx] = ux[idx];

		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>1 && i<NX - 1 /*&& j>0 && j<N-1 && k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) != sweepflag && mark(i - 1, j, k) != sweepflag))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (j + dj<0 || j + dj>NY - 1 || k + dk<0 || k + dk >NZ -1)
					continue;
				wx = -di*(ls(i, j, k) - ls(i - 1, j, k));
				if (wx<0)
					continue;
				wy = (ls(i, j, k) + ls(i - 1, j, k) - ls(i, j + dj, k) - ls(i - 1, j + dj, k))*0.5f;
				if (wy<0)
					continue;
				wz = (ls(i, j, k) + ls(i - 1, j, k) - ls(i, j, k + dk) - ls(i - 1, j, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outux(i, j, k) = wx*ux(i + di, j, k) + wy* ux(i, j + dj, k) + wz* ux(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//copy
		outuy[idx] = uy[idx];

		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if ( /*i>0 && i<N-1 &&*/ j>1 && j<NY - 1 /*&& k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) != sweepflag && mark(i, j - 1, k) != sweepflag))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di>NX - 1 || k + dk<0 || k + dk >NZ - 1)
					continue;
				wy = -dj*(ls(i, j, k) - ls(i, j - 1, k));
				if (wy<0)
					continue;
				wx = (ls(i, j, k) + ls(i, j - 1, k) - ls(i + di, j, k) - ls(i + di, j - 1, k))*0.5f;
				if (wx<0)
					continue;
				wz = (ls(i, j, k) + ls(i, j - 1, k) - ls(i, j, k + dk) - ls(i, j - 1, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuy(i, j, k) = wx*uy(i + di, j, k) + wy* uy(i, j + dj, k) + wz* uy(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//copy
		outuz[idx] = uz[idx];

		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if ( /*i>0 && i<N-1 && j>0 && j<N-1 &&*/ k>1 && k<NZ - 1)
		{
			if ((mark(i, j, k) != sweepflag && mark(i, j, k - 1) != sweepflag))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di >NX - 1 || j + dj<0 || j + dj>NY - 1)
					continue;
				wz = -dk*(ls(i, j, k) - ls(i, j, k - 1));
				if (wz<0)
					continue;
				wy = (ls(i, j, k) + ls(i, j, k - 1) - ls(i, j + dj, k) - ls(i, j + dj, k - 1))*0.5f;
				if (wy<0)
					continue;
				wx = (ls(i, j, k) + ls(i, j, k - 1) - ls(i + di, j, k) - ls(i + di, j, k - 1))*0.5f;
				if (wx<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuz(i, j, k) = wx*uz(i + di, j, k) + wy* uz(i, j + dj, k) + wz* uz(i, j, k + dk);
			}
		}
	}
}


__global__ void correctbubblepos(farray ls, farray phigrax, farray phigray, farray phigraz, float3 *ppos, char* pflag, int pnum, float *pphi)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		char iflag = pflag[idx];
		//test. todo. debug
		if (iflag == TYPEAIRSOLO || iflag == TYPESOLID)
			return;

		float3 ipos = ppos[idx];
		int s = (iflag == TYPEFLUID) ? -1 : 1;
		float d, dirlen, rs = 0.5f*dparam.cellsize.x;
		float3 dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
		dirlen = length(dir);
		if (dirlen == 0)
			return;
		else
			dir = normalize(dir);
		d = getScaleFromFrid(ipos, ls) / dirlen;
		//test
		// 		if( s*d<0 )
		// 			ipos=ipos +rs*dir;
		//debug.
		pphi[idx] = d;

		//todo: 这里有问题
		if (s*d<0 && abs(d)<0.5f*dparam.cellsize.x)	//wrong way
		{
			if (iflag == TYPEAIR&& abs(d)>0.3f*dparam.cellsize.x)	//
			else if (iflag == TYPEFLUID)
			{
				ipos = ipos - d*dir;

				dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
				dirlen = length(dir);
				if (dirlen == 0)
					return;
				else
					dir = normalize(dir);
				d = getScaleFromFrid(ipos, ls) / dirlen;

				ipos = ipos + s*(rs - s*d)*dir;
			}
			//	cnt++;
		}
		else if (iflag == TYPEFLUID && s*d<rs*0.5f && s*d >= 0)		//todo: rs*0.5f有点小问题，但不加这个0.5的话流体的体积会变化明显
		{
			ipos = ipos + s*(rs - s*d)*dir;
		}
		ppos[idx] = ipos;
	}
}

__global__ void correctbubblepos_air(farray lsmerge, farray phigrax, farray phigray, farray phigraz, farray lsair, farray phigrax_air, farray phigray_air, farray phigraz_air, float3 *ppos, char* pflag, int pnum, float *pphi)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		char iflag = pflag[idx];
		//test. todo. debug
		if (iflag == TYPEAIRSOLO || iflag == TYPESOLID)
			return;

		float3 ipos = ppos[idx];
		int s = (iflag == TYPEFLUID) ? -1 : 1;
		float d, dirlen, rs = 0.5f*dparam.cellsize.x;
		float3 dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
		dirlen = length(dir);
		if (dirlen == 0)
			return;
		else
			dir = normalize(dir);
		d = getScaleFromFrid(ipos, lsmerge) / dirlen;
		//test
		// 		if( s*d<0 )
		// 			ipos=ipos +rs*dir;
		//debug.
		pphi[idx] = d;

		//todo: 这里有问题
		if (s*d<0 && abs(d)<0.5f*dparam.cellsize.x)	//wrong way
		{
			if (iflag == TYPEAIR&& abs(d)>0.3f*dparam.cellsize.x)	//气体粒子只在错位比较明显的情况下才纠正，主要是为了防止气泡体积的收缩。
				ipos = ipos - d*dir;

			//	cnt++;
		}
		if (iflag == TYPEFLUID)	//对液体粒子使用气体的level set来处理，慢慢把液体“挤出”气泡之外，使得lsmerge计算更为准确
		{
			dir = getVectorFromGrid(ipos, phigrax_air, phigray_air, phigraz_air);
			dirlen = length(dir);
			if (dirlen == 0)
				return;
			else
				dir = normalize(dir);
			d = getScaleFromFrid(ipos, lsair) / dirlen;

			if (d<-1.3f*rs)
				ipos = ipos - (d - rs)*dir;
		}

		ppos[idx] = ipos;
	}
}

//根据levelset计算梯度场，相当于一个方向
__global__ void computePhigra(farray phigrax, farray phigray, farray phigraz, farray ls)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float h = dparam.cellsize.x;
		float lsx1, lsx0, lsy1, lsy0, lsz1, lsz0, lscenter = ls[idx];
		lsx1 = (verifycellidx(i + 1, j, k)) ? ls(i + 1, j, k) : lscenter;
		lsx0 = (verifycellidx(i - 1, j, k)) ? ls(i - 1, j, k) : lscenter;
		lsy1 = (verifycellidx(i, j + 1, k)) ? ls(i, j + 1, k) : lscenter;
		lsy0 = (verifycellidx(i, j - 1, k)) ? ls(i, j - 1, k) : lscenter;
		lsz1 = (verifycellidx(i, j, k + 1)) ? ls(i, j, k + 1) : lscenter;
		lsz0 = (verifycellidx(i, j, k - 1)) ? ls(i, j, k - 1) : lscenter;

		//todo: 这里需要考虑一下
		phigrax[idx] = ((lsx1 - lsx0)*0.5f) / h;
		phigray[idx] = ((lsy1 - lsy0)*0.5f) / h;
		phigraz[idx] = ((lsz1 - lsz0)*0.5f) / h;

		//phigrax[idx] = (lsx1-lscenter)/h;
		//phigray[idx] = (lsy1-lscenter)/h;
		//phigraz[idx] = (lsz1-lscenter)/h;
	}
}

__global__ void copyParticle2GL_phi(float3* ppos, char *pflag, float *pmass, float *pTemperature, int pnum, float *renderpos, float *rendercolor,
	farray ls, farray phigrax, farray phigray, farray phigraz, char typeflag, float Tmax, float Tmin)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//todo:
		if (pflag[idx] == typeflag/* || ppos[idx].y<NY*0.5f*dparam.cellsize.x */)
		{
			renderpos[idx * 3] = -2.0f;
			renderpos[idx * 3 + 1] = 0.0f;
			renderpos[idx * 3 + 2] = 0.0f;

			float3 color = make_float3(0.0f);
			rendercolor[idx * 3] = color.x;
			rendercolor[idx * 3 + 1] = color.y;
			rendercolor[idx * 3 + 2] = color.z;
			return;
		}
		renderpos[idx * 3] = ppos[idx].x;
		renderpos[idx * 3 + 1] = ppos[idx].y;
		renderpos[idx * 3 + 2] = ppos[idx].z;

		float3 color;

		if (pflag[idx] == TYPEAIR)
			color = mapColorBlue2Red(0.0f);
		else if (pflag[idx] == TYPEFLUID)
			color = mapColorBlue2Red(2.0f);
		else if (pflag[idx] == TYPESOLID)
			color = mapColorBlue2Red(4.0f);
		else
			color = mapColorBlue2Red(6.0f);
		//color=mapColorBlue2Red( (pTemperature[idx]-Tmin)/(Tmax-Tmin)*6.0f );


		rendercolor[idx * 3] = color.x;
		rendercolor[idx * 3 + 1] = color.y;
		rendercolor[idx * 3 + 2] = color.z;
	}
}

//压强与速度的计算，加入surface tension. [2005]Discontinuous Fluids
__global__ void subGradPress_bubble(farray p, farray ux, farray uy, farray uz, farray sf, farray lsmerge, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float h = dparam.cellsize.x;
	float J = 0.0f, theta;
	if (idx<dparam.gvnum.x)
	{
		J = 0.0f;
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>0 && i<NX)		//look out for this condition
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID) || (mark(i, j, k) == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR))
			{
				theta = (0.0f - lsmerge(i - 1, j, k)) / (lsmerge(i, j, k) - lsmerge(i - 1, j, k));
				J = theta*sf(i - 1, j, k) + (1.0f - theta)*sf(i, j, k);
			}
			ux(i, j, k) -= (p(i, j, k) - p(i - 1, j, k) - J) / h;
		}
	}
	if (idx<dparam.gvnum.y)
	{
		J = 0.0f;
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if (j>0 && j<NY)		//look out for this condition
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID) || (mark(i, j, k) == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR))
			{
				theta = (0.0f - lsmerge(i, j - 1, k)) / (lsmerge(i, j, k) - lsmerge(i, j - 1, k));
				J = theta*sf(i, j - 1, k) + (1.0f - theta)*sf(i, j, k);
			}
			uy(i, j, k) -= (p(i, j, k) - p(i, j - 1, k) - J) / h;
		}
	}
	if (idx<dparam.gvnum.z)
	{
		J = 0.0f;
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if (k>0 && k<NZ)		//look out for this condition
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID) || (mark(i, j, k) == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR))
			{
				theta = (0.0f - lsmerge(i, j, k - 1)) / (lsmerge(i, j, k) - lsmerge(i, j, k - 1));
				J = theta*sf(i, j, k - 1) + (1.0f - theta)*sf(i, j, k);
			}
			uz(i, j, k) -= (p(i, j, k) - p(i, j, k - 1) - J) / h;
		}
	}
}

__global__ void sweepVacuum(charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (mark[idx] != TYPEAIR)
			return;
		//mark
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		if (mark(i + di, j + dj, k + dk) == TYPEVACUUM)
			mark[idx] = TYPEVACUUM;

	}
}

__global__ void markDeleteAirParticle(float3* ppos, char* pflag, float *pmass, uint *preservemark, int pnum, charray mark, farray lsmerge, farray lsair, uint *cnt)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		//fluid and solid particles are preserved, air and airsolo particles are verified.
		if (pflag[idx] == TYPESOLID)
		{
			preservemark[idx] = 1;
			return;
		}

		int i, j, k;
		getijkfrompos(i, j, k, ppos[idx]);

		if (pflag[idx] == TYPEFLUID)
		{
			float lsm = getScaleFromFrid(ppos[idx], lsmerge);
			float lsa = getScaleFromFrid(ppos[idx], lsair);
			if ( /*lsm>1.2f*dparam.cellsize.x || */lsa<-1.0*dparam.cellsize.x)
				preservemark[idx] = 0, cnt[0]++;
			else
				preservemark[idx] = 1;
			return;
		}

		int cnt = 0;
		for (int di = -1; di <= 1; di += 1) for (int dj = -1; dj <= 1; dj += 1) for (int dk = -1; dk <= 1; dk += 1)
		if (verifycellidx(i + di, j + dj, k + dk) && mark(i + di, j + dj, k + dk) == TYPEVACUUM)
			cnt++;
		if (cnt == 0 && pmass[idx]>0.000001f)		//notice: 这里附带的删除了质量过小的气体粒子，与气体粒子的被吸收有关
			preservemark[idx] = 1;
		else
			preservemark[idx] = 0;
	}
}

// compact voxel array
__global__ void deleteparticles(uint *preserveflag, uint *preserveflagscan, int pnum, float3 *outpos, float3 *pos,
	float3 *outvel, float3 *vel, float *outmass, float* mass, char *outflag, char *flag, float *outTemperature, float *temperature, float *outheat, float *heat,
	float *outsolubility, float *solubility, float *outgascontain, float *gascontain)
{
	uint idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	if (idx<pnum)
	{
		if (preserveflag[idx] == 1)
		{
			//deleteflagscan 存的是删除某些粒子之后的"索引".
			uint outidx = preserveflagscan[idx];
			outpos[outidx] = pos[idx];
			outvel[outidx] = vel[idx];
			outmass[outidx] = mass[idx];
			outflag[outidx] = flag[idx];
			outTemperature[outidx] = temperature[idx];
			outheat[outidx] = heat[idx];
			outsolubility[outidx] = solubility[idx];
			outgascontain[outidx] = gascontain[idx];
		}
	}
}

__device__ int cntairparticle(float3 *ppos, char *pflag, int igrid, uint *gridstart, uint *gridend, const float3 &ipos, float r)
{
	uint start = gridstart[igrid];
	int res = 0;
	float dis;
	if (start == CELL_UNDEF)
		return res;
	for (int p = start; p<gridend[igrid]; p++)
	{
		dis = length(ppos[p] - ipos);
		if (dis<r && (pflag[p] == TYPEAIR || pflag[p] == TYPEAIRSOLO))
		{
			++res;
		}
	}
	return res;
}

__device__ inline bool isInBoundaryCell(int x, int y, int z)
{
	int level = 2;
	if (x <= level || x >= NX - 1 - level || y <= level || y >= NY - 1 - level)
		return true;
	else
		return false;
}

__global__ void verifySoloAirParticle(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray lsmerge, farray airux, farray airuy, farray airuz, uint *gridstart, uint *gridend, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		char iflag = pflag[idx];
		if (iflag == TYPEFLUID || iflag == TYPESOLID)	//TYPEAIR, TYPEAIRSOLO can go on.
			return;

		float3 ipos = ppos[idx];
		float ls = getScaleFromFrid(ipos, lsmerge);
		float h = dparam.cellsize.x;
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		//a key adjustment, the tolerent will affect the result directly.
		int cnt = 0;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			cnt += cntairparticle(ppos, pflag, getidx(i + di, j + dj, k + dk), gridstart, gridend, ipos, h);

		float tol1 = -1.45f, tol2 = -0.5f;
		if (scene == SCENE_MELTANDBOIL || scene == SCENE_MELTANDBOIL_HIGHRES || scene==SCENE_ALL)
			tol1 = 0.05f, tol2 = -0.8f;
		else if (scene == SCENE_INTERACTION)
			tol1 = 0.2f, tol2 = -0.5f;

		if ((cnt >= 10 || ls>tol1*h) && pflag[idx] == TYPEAIRSOLO && !isInBoundaryCell(i, j, k))		//decide whether the air solo particle should  be transfered to air particle.
		{
			if (cnt >= 3)
				pflag[idx] = TYPEAIR;
		}
		else if (iflag == TYPEAIR && (isInBoundaryCell(i, j, k) || ls<tol2*h || cnt <= 1))
		{
			//todo: 插值速度 or not???
			//pvel[idx]= pvel[idx]*0.8f + 0.2f*getParticleVelFromGrid(ipos,airux,airuy,airuz);
			pvel[idx] = getParticleVelFromGrid(ipos, airux, airuy, airuz);
			pflag[idx] = TYPEAIRSOLO;
		}
	}
}

__device__ float sumdensity(float3 ipos, float h2, int grididx, float3 *ppos, char *pflag, uint *gridstart, uint *gridend)
{
	float res = 0;
	uint start = gridstart[grididx];
	if (start == CELL_UNDEF)
		return res;
	float dist2;
	for (uint p = start; p<gridend[grididx]; p++)
	{
		// notice: should include liquid particle, not just spray particle.
		if (pflag[p] != TYPEAIR && pflag[p] != TYPEAIRSOLO)
			continue;
		dist2 = dot(ppos[p] - ipos, ppos[p] - ipos);
		if (dist2<h2)
			res += pow(h2 - dist2, 3.0f);	//todo: m0 or pmass[p]?
	}
	return res;
}

__global__ void calcDensPress_Air(float3* ppos, float *pdens, float *ppress, char* pflag, int pnum, uint *gridstart, uint *gridend)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] != TYPEAIR && pflag[idx] != TYPEAIRSOLO)
			return;

		float3 ipos = ppos[idx];
		float h = dparam.cellsize.x;		//todo: set support radius, key part.
		float h2 = h*h;
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		float dens = 0;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			dens += sumdensity(ipos, h2, getidx(i + di, j + dj, k + dk), ppos, pflag, gridstart, gridend);

		dens *= dparam.airm0 * dparam.poly6kern;
		if (dens == 0) dens = 1.0f;
		pdens[idx] = 1.0f / dens;
		ppress[idx] = 1.5f * (dens - dparam.waterrho*0.5f);
	}
}


__device__ float3 sumforce(float3 *ppos, float3 *pvel, float *ppress, float *pdens, char *pflag, int grididx, uint *gridstart, uint *gridend,
	float3 ipos, float3 ivel, float ipress, float idens, float h, float kvis)
{
	uint start = gridstart[grididx];
	float3 res = make_float3(0.0f), dir;
	float dis, c, pterm, dterm;// kattrct=0.0f, 
	if (start == CELL_UNDEF)
		return res;
	float vterm = dparam.lapkern * kvis;

	for (uint p = start; p<gridend[grididx]; p++)
	{
		dir = ipos - ppos[p];
		dis = length(dir);
		if (dis>0 && dis<h && (pflag[p] == TYPEAIRSOLO || pflag[p] == TYPEAIR))
		{
			c = h - dis;
			pterm = -0.5f * c * dparam.spikykern * (ipress + ppress[p]) / dis;
			dterm = c * idens * pdens[p];
			res += (pterm * dir + vterm * (pvel[p] - ivel)) * dterm;
		}
	}
	return res;
}
__global__ void enforceForceSoloAirP(float3 *ppos, float3 *pvel, float *pdens, float *ppress, char *pflag, int pnum, uint *gridstart, uint *gridend, float viscositySPH, float maxVelForBubble)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] != TYPEAIRSOLO && pflag[idx] != TYPEAIR)
			return;

		float3 ipos = ppos[idx];
		float3 ivel = pvel[idx];
		float ipress = ppress[idx], idens = pdens[idx];
		float h = dparam.cellsize.x;
		//float kvis=0.0f;	

		int i, j, k;
		float3 force = make_float3(0.0f);
		getijkfrompos(i, j, k, ipos);

		int width = 1;
		for (int di = -width; di <= width; di++) for (int dj = -width; dj <= width; dj++) for (int dk = -width; dk <= width; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			force += sumforce(ppos, pvel, ppress, pdens, pflag, getidx(i + di, j + dj, k + dk), gridstart, gridend, ipos, ivel, ipress, idens, h, viscositySPH);

		//todo: 直接更新速度和位置??
		force *= dparam.airm0;
		//force = make_float3(0);

		ivel += force*dparam.dt;
		ipos += ivel*dparam.dt;

		//restrict the vel below a threshold.
		// 		if( length(ivel) > maxVelForBubble )
		// 			ivel = normalize(ivel) * maxVelForBubble;
		// 	
		//	advect particle, using rho!!!!
		//	ppos[idx]=ipos;
		pvel[idx] = ivel;
	}
}


__device__ float sumdensity_SLCouple(float3 ipos, float h2, int grididx, float3 *ppos, char *pflag, uint *gridstart, uint *gridend)
{
	float res = 0;
	uint start = gridstart[grididx];
	if (start == CELL_UNDEF)
		return res;
	float dist2;
	for (uint p = start; p<gridend[grididx]; p++)
	{
		dist2 = dot(ppos[p] - ipos, ppos[p] - ipos);
		if (dist2<h2)
			res += pow(h2 - dist2, 3.0f);
	}
	return res;
}

//solid-liquid coupling, in SPH framework
__global__ void calcDensPressSPH_SLCouple(float3* ppos, float *pdens, float *ppress, char* pflag, int pnum, uint *gridstart, uint *gridend)
{
	
}

__device__ float3 sumforce_SLCouple(float3 *ppos, float3 *pvel, float *ppress, float *pdens, char *pflag, int grididx, uint *gridstart, uint *gridend,
	float3 ipos, float3 ivel, float ipress, float idens, float h, float kvis)
{
	uint start = gridstart[grididx];
	float3 res = make_float3(0.0f), dir;
	float dis, c, pterm, dterm;// kattrct=0.0f, kvis=0.0f;
	if (start == CELL_UNDEF)
		return res;
	float vterm = dparam.lapkern * kvis;

	for (uint p = start; p<gridend[grididx]; p++)
	{
		dir = ipos - ppos[p];
		dis = length(dir);
		if (dis>0 && dis<h)
		{
			c = h - dis;
			pterm = -0.5f * c * dparam.spikykern * (ipress + ppress[p]) / dis;
			dterm = c * idens * pdens[p];
			res += (pterm * dir + vterm * (pvel[p] - ivel)) * dterm;
		}
	}
	return res;
}
__global__ void enforceForceSPH_SLCouple(float3 *ppos, float3 *pvel, float *pdens, float *ppress, char *pflag, int pnum, uint *gridstart, uint *gridend, float viscositySPH)
{
	
}

__global__ void updateFixedHeat(farray fixedHeat, int frame)
{
	
}

__global__ void addHeatAtBottom(farray Tp, int frame, float heatIncreaseBottom)
{
	
}

//
__global__ void compb_heat(farray Tp_old, farray Tp, farray fixedheat, charray mark, float *heatAlphaArray)
{
	
}

//z = Ax: A is a sparse matrix, representing the left hand item of Poisson equation.
__global__ void computeAx_heat(farray ans, charray mark, farray x, int n, float *heatAlphaArray, farray fixedHeat, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		float h = dparam.cellsize.x;
		float dt = dparam.dt;

		float alpha = heatAlphaArray[mark[idx]];

		if (mark[idx] != TYPEBOUNDARY/* && mark[idx]!=TYPEVACUUM*/)
		{
			int i, j, k;
			getijk(i, j, k, idx);

			float center = x[idx];
			float sum = (h*h / alpha / dt + 6.0f)*center;

			//trick: 决定要不要让freeair参与计算
			if (scene == SCENE_BOILING || scene == SCENE_BOILING_HIGHRES || scene == SCENE_MELTANDBOIL || scene == SCENE_MELTANDBOIL_HIGHRES || scene ==SCENE_ALL)
			{
				sum -= ((mark(i + 1, j, k) == TYPEBOUNDARY || mark(i + 1, j, k) == TYPEVACUUM) ? center : x(i + 1, j, k));
				sum -= ((mark(i, j + 1, k) == TYPEBOUNDARY || mark(i, j + 1, k) == TYPEVACUUM) ? center : x(i, j + 1, k));
				sum -= ((mark(i, j, k + 1) == TYPEBOUNDARY || mark(i, j, k + 1) == TYPEVACUUM) ? center : x(i, j, k + 1));
				sum -= ((mark(i - 1, j, k) == TYPEBOUNDARY || mark(i - 1, j, k) == TYPEVACUUM) ? center : x(i - 1, j, k));
				sum -= ((mark(i, j - 1, k) == TYPEBOUNDARY || mark(i, j - 1, k) == TYPEVACUUM) ? center : x(i, j - 1, k));
				sum -= ((mark(i, j, k - 1) == TYPEBOUNDARY || mark(i, j, k - 1) == TYPEVACUUM) ? center : x(i, j, k - 1));
			}
			else
			{
				sum -= ((mark(i + 1, j, k) == TYPEBOUNDARY) ? center : x(i + 1, j, k));
				sum -= ((mark(i, j + 1, k) == TYPEBOUNDARY) ? center : x(i, j + 1, k));
				sum -= ((mark(i, j, k + 1) == TYPEBOUNDARY) ? center : x(i, j, k + 1));
				sum -= ((mark(i - 1, j, k) == TYPEBOUNDARY) ? center : x(i - 1, j, k));
				sum -= ((mark(i, j - 1, k) == TYPEBOUNDARY) ? center : x(i, j - 1, k));
				sum -= ((mark(i, j, k - 1) == TYPEBOUNDARY) ? center : x(i, j, k - 1));
			}

			ans[idx] = sum;
		}
	}
}


//Ans = x + a*y
__global__ void pcg_op_heat(charray A, farray ans, farray x, farray y, float a, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		//	if( A[idx]==TYPEFLUID || A[idx]==TYPEAIR )
		if (A[idx] != TYPEBOUNDARY)
			ans[idx] = x[idx] + a*y[idx];
		else
			ans[idx] = 0.0f;
	}
}


__global__ void setBoundaryHeat(farray tp)
{
	
}

__global__ void compTpChange(farray tp, farray tpsave, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] != TYPEBOUNDARY)
			tpsave[idx] = tp[idx] - tpsave[idx];
		else
			tpsave[idx] = 0;
	}
}

__device__ void sumHeat(float &heatsum, float &weight, float3 gpos, float3 *pos, float *pTemperature,
	uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
		w = sharp_kernel(dis2, RE);
		weight += w;
		heatsum += w*pTemperature[p];
	}
}

__global__ void mapHeatp2g_hash(float3 *ppos, float *pTemperature, int pnum, farray heat, uint* gridstart, uint *gridend, float defaulttemperature)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		float weight = 0.0f, heatsum = 0;
		float3 gpos;
		getijk(i, j, k, idx);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumHeat(heatsum, weight, gpos, ppos, pTemperature, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		heatsum = (weight>0) ? (heatsum / weight) : defaulttemperature;
		heat(i, j, k) = heatsum;
	}
}

__global__ void mapHeatg2p(float3 *ppos, char *parflag, float *pTemperature, int pnum, farray Tchange, farray T, float defaultSolidT, float alphaTempTrans)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		pTemperature[idx] = alphaTempTrans*(pTemperature[idx] + getScaleFromFrid(ipos, Tchange)) + (1 - alphaTempTrans)*getScaleFromFrid(ipos, T);		//use a scheme like FLIP, update the particle temperature by heat change.
	}
}

__global__ void mapHeatg2p_MeltAndBoil(float3 *ppos, char *parflag, float *pTemperature, int pnum, farray Tchange, farray T, float defaultSolidT, float alphaTempTrans)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		float newtemp = alphaTempTrans*(pTemperature[idx] + getScaleFromFrid(ipos, Tchange)) + (1 - alphaTempTrans)*getScaleFromFrid(ipos, T);		//use a scheme like FLIP, update the particle temperature by heat change.
		if (parflag[idx] == TYPESOLID)
			pTemperature[idx] = 0.95f*(pTemperature[idx]) + 0.05f*newtemp;
		else
			pTemperature[idx] = newtemp;
	}
}

__global__ void initHeatParticle(float *pTemperature, float *pHeat, float defaultSolidT, float defaultLiquidT, float LiquidHeatTh, char *pflag, int pnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPESOLID)
		{
			pTemperature[idx] = defaultSolidT;
			pHeat[idx] = 0;
		}
		else
		{
			pTemperature[idx] = defaultLiquidT;
			pHeat[idx] = LiquidHeatTh;
		}
	}
}

//Temperature0=273.15K, Solubility0=1.0f 
__global__ void initsolubility_k(float *psolubility, float* pgascontain, float *ptemperature, char *pflag, int pnum, float Solubility0, float Temperature0, float dissolvegasrate, float initgasrate)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPEFLUID || pflag[idx] == TYPESOLID)
		{
			psolubility[idx] = dissolvegasrate*dparam.airm0 * exp(1018.9f*(1 / ptemperature[idx] - 1 / Temperature0));	//todo: adjust the parameter.
			pgascontain[idx] = initgasrate*psolubility[idx];
		}
		else
		{
			psolubility[idx] = 0;
			pgascontain[idx] = 0;
		}
	}
}

//Temperature0=273.15K, Solubility0=1.0f (每1个流体粒子里含的气体够生成一个完事的气体粒子)
__global__ void updatesolubility(float *psolubility, float *ptemperature, char *pflag, int pnum, float Solubility0, float Temperature0, float dissolvegasrate)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPEFLUID)
			psolubility[idx] = dissolvegasrate*dparam.airm0 * exp(1018.9f*(1 / ptemperature[idx] - 1 / Temperature0));	//todo: adjust the parameter.
	}
}

//addparnums初始化应该是0
__global__ void GenerateGasParticle_k(float *psolubility, float *paircontain, float3 *ppos, float3 *pvel, float *pmass, char *pflag, float *pTemperature, float *pLHeat,
	int pnum, uint *gridstart, uint *gridend, int *addparnums, float *randfloat, int randcnts, int frame, farray gTemperature, float LiquidHeatTh,
	int *seedcell, int seednum, float vaporGenRate)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		float gcontain = 0, gsolubility = 0, gairexist = 0;
		int liquidParCnt = 0, gasParCnt = 0;
		float airparticlemass0 = dparam.airm0;		//todo
		float vaporsum = 0;//, vaporrate = 0.01f;
		float3 gaspos = make_float3(0), gasvel = make_float3(0);
		int i, j, k;
		getijk(i, j, k, idx);

		if (k <= 1 || isInBoundaryCell(i, j, k))	return;		//最下面的一行不生成气泡粒子

		float3 gpos = make_float3(i, j, k)*dparam.cellsize.x;

		uint start = gridstart[idx];
		if (start == CELL_UNDEF)
			return;
		//1. 统计气体含量、流体粒子含有的气体量、可溶解量
		for (int p = start; p<gridend[idx]; p++)
		{
			if (pflag[p] == TYPEFLUID)
			{
				gcontain += paircontain[p];
				gsolubility += psolubility[p];

				vaporsum += max(0.0f, pLHeat[p] - LiquidHeatTh) * vaporGenRate * airparticlemass0;

				liquidParCnt++;
			}
			else if (pflag[p] == TYPEAIRSOLO || pflag[p] == TYPEAIR)
			{
				gairexist += pmass[p];
				gaspos += ppos[p];
				gasvel += pvel[p];
				gasParCnt++;
			}
		}

		bool hasseed = false;
		for (int i = 0; i<seednum; i++)
		if (seedcell[i] == idx) hasseed = true;

		//如有必要，增加一个气体粒子
		int addcnt = 0;
		int randbase = (idx*frame) % (randcnts - 200);
		//randpos and randfloat are in [0,1]
		float3 randpos = make_float3(randfloat[(randbase + addcnt++) % randcnts], randfloat[(randbase + addcnt++) % randcnts], randfloat[(randbase + addcnt++) % randcnts]);
		float randnum = randfloat[(randbase + addcnt++) % randcnts];
		float r = dparam.cellsize.x * 0.25f;
		if (gcontain - gsolubility + vaporsum > airparticlemass0 && (hasseed || gasParCnt>0))
		{
			int addindex = atomicAdd(&addparnums[0], 1) + pnum;
			pmass[addindex] = airparticlemass0;//dparam.m0;	//todo:
			if (gasParCnt>0)
			{
				ppos[addindex] = gaspos / gasParCnt + (max(0.5f, randnum)*r) * (randpos - make_float3(0.5f)) * 2;	//与凝结核有关
				pvel[addindex] = make_float3(0.0f);//gasvel/gasParCnt;			//与已有的气体粒子有关	
			}
			else
			{
				ppos[addindex] = gpos + dparam.cellsize.x*randpos;
				pvel[addindex] = make_float3(0.0f);
			}
			pflag[addindex] = TYPEAIRSOLO;
			pTemperature[addindex] = gTemperature[idx];		//网格温度
			pLHeat[addindex] = 0;		//气体粒子的heat无所谓
			paircontain[addindex] = 0.0f;
			psolubility[addindex] = 0.0f;

			//重置液体粒子的气体含量
			for (int p = start; p<gridend[idx]; p++)
			{
				if (pflag[p] == TYPEFLUID)
				{
					paircontain[p] = min(paircontain[p], psolubility[p]);
					pLHeat[p] = min(pLHeat[p], LiquidHeatTh);
					//todo: decrease the liquids mass.
				}
			}
		}

	}
}

//addparnums初始化应该是0
__global__ void updatebubblemass(float *psolubility, float *paircontain, float3 *ppos, float *pmass, char *pflag, int pnum, uint *gridstart, uint *gridend)
{
	
}

//使用预计算好的位置根据温度和溶解度生成empty气泡，当气泡大于一定体积时，生成AIR粒子。
//对其它模块的影响：markgrid, correctpos, heattransfer.
__global__ void updateEmptyBubbles(float3 *pepos, float3 *pedir, float *peradius, int penum, float3 *parpos, float3 *parvel, float *parmass, float* parTemperature,
	char *parflag, float *parsolubility, float *paraircontain, int parnum, int *addparnums, uint *gridstart, uint *gridend, farray gTemperature)
{
	
}

__device__ void mat4_mul(matrix4* dst, const matrix4* m0, const matrix4* m1)
{
	int row;
	int col;
	int i;


	for (row = 0; row < 4; row++)
	for (col = 0; col < 4; col++)
	for (i = 0; i < 4; i++)
		dst->m[row * 4 + col] += m0->m[row * 4 + i] * m1->m[i * 4 + col];

}
__device__ void mat4_mulvec3_as_mat3(float3* dst, const matrix4* m, const float3* v)
{
	float new_x;
	float new_y;
	float new_z;

	new_x = v->x*m->m[0 + 4 * 0] + v->y*m->m[0 + 4 * 1] + v->z*m->m[0 + 4 * 2];
	new_y = v->x*m->m[1 + 4 * 0] + v->y*m->m[1 + 4 * 1] + v->z*m->m[1 + 4 * 2];
	new_z = v->x*m->m[2 + 4 * 0] + v->y*m->m[2 + 4 * 1] + v->z*m->m[2 + 4 * 2];
	dst->x = new_x;
	dst->y = new_y;
	dst->z = new_z;
}

__global__ void MeltingSolidByHeat(float *pTemperature, float *pLHeat, char *pflag, int pnum, float LiquidHeatTh, float meltTemperature, int *numchange)
{
	
}

__global__ void FreezingSolidByHeat(float3* ppos, float *pLHeat, char *pflag, int pnum, int *numchange, uint *gridstart, uint *gridend)
{
	
}

//计算air solo particle与流体场之间的drag force，直接在本函数里修改了速度。以dragparam为影响大小的参数。
__global__ void calDragForce(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray ux, farray uy, farray uz, float dragparamsolo, float dragparamgrid, SCENE scene)
{
	
}

__global__ void accumulate_GPU_k(int num, float3* out, float3* a)//dsum, a.data, flag, n
{
	extern __shared__ float3 ddata[];

	uint tid = threadIdx.x;
	uint i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	ddata[tid] = (i >= num) ? make_float3(0, 0, 0) : a[i];	//赋值给solidparticles
	__syncthreads();

	for (int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid<s)
			ddata[tid] += ddata[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		out[blockIdx.x] = ddata[0];
}

__global__ void accumulate_GPU_k(int num, float3* out, float3* a, float* b)//dsum, a.data, flag, n
{
	extern __shared__ float3 ddata[];

	uint tid = threadIdx.x;
	uint i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	ddata[tid] = (i >= num) ? make_float3(0, 0, 0) : a[i]*b[i];	//赋值给solidparticles
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			ddata[tid] += ddata[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		out[blockIdx.x] = ddata[0];
}

__global__ void accumulate_GPU_k(int num, float3* out, float3* a, float3* b)
{
	extern __shared__ float3 ddata[];

	uint tid = threadIdx.x;
	uint i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	ddata[tid] = (i >= num) ? make_float3(0, 0, 0) : a[i]*b[i];	//赋值给solidparticles
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			ddata[tid] += ddata[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		out[blockIdx.x] = ddata[0];
}

__global__ void accumulate_GPU_k_float(int num, float* out, float* a)//dsum, a.data, flag, n
{
	extern __shared__ float fddata[];

	uint tid = threadIdx.x;
	uint i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	fddata[tid] = (i >= num) ? 0 : a[i];	//赋值给solidparticles
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			fddata[tid] += fddata[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		out[blockIdx.x] = fddata[0];
}



__global__ void compute_cI_k(int pnum, char* parflag, float3 *parPos, float3 *parVel, float3* c, float3* weight, float3 rg)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	if (idx<pnum)
	{
		if (parflag[idx] == TYPESOLID)
		{
			float dis = length(parPos[idx] - rg);
			if (dis>1e-6)
			{
				c[idx] = cross(parPos[idx] - rg, parVel[idx]);
				weight[idx] = make_float3(dis, 0, 0);
			}
			else
				c[idx] = weight[idx] = make_float3(0);
		}
		else
		{
			c[idx] = weight[idx] = make_float3(0);
			//c[idx] = make_float3(0,0,0);
		}
	}
}

__global__ void setVelZeroSolid_k(float3 *parvel, char *parflag, int pnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum && parflag[idx] == TYPESOLID)
		parvel[idx] = make_float3(0);
}

__global__ void computeVelSolid_k(float3* parPos, char* parflag, float3* parVel, int pnum, float3 rg, float3 R, float3 T)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum && parflag[idx] == TYPESOLID)
	{

		float3 v_half = cross(R, parPos[idx] - rg);		//粒子的角速度`
		v_half += T;								//固体粒子的总速度
		v_half = 0.5*(parVel[idx] + v_half);
		parVel[idx] = v_half;
		//	parVel[idx] = make_float3(0);
	}
}

__device__ inline float3 transposeParticle(float3 p, matrix3x3 rm)
{
	float3 res;
	res.x = p.x*rm.x00 + p.y*rm.x10 + p.z*rm.x20;
	res.y = p.x*rm.x01 + p.y*rm.x11 + p.z*rm.x21;
	res.z = p.x*rm.x02 + p.y*rm.x12 + p.z*rm.x22;
	return res;
}
//由rotation matrix "rm"来计算各粒子的位置
__global__ void computePosSolid_k(float3* parvel, float3* parPos, char* parflag, int pnum, float3 rg, float3 rg0, matrix3x3 rm)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && parflag[idx] == TYPESOLID)
	{
		float3 transp = parPos[idx] - rg0;
		transp = transposeParticle(transp, rm);
		parPos[idx] = transp + rg;
		//if (length(parPos[idx])<10.5)
		//parPos[idx] -= parvel[idx] * 0.00001;
	
	}
}

__global__ void computeSolidVertex_k(float3* vertexpos, int vnum, float3 rg, float3 rg0, matrix3x3 rm)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<vnum)
	{
		float3 transp = vertexpos[idx] - rg0;
		transp = transposeParticle(transp, rm);
		vertexpos[idx] = transp + rg;
	}
}

__global__ void set_nonsolid_2_zero(char* pflag, int pnum, float3* Pos, float3* Vel)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && pflag[idx] != TYPESOLID)
	{
		Pos[idx] = make_float3(0, 0, 0);
		Vel[idx] = make_float3(0, 0, 0);
		//Mass[idx] = 0.;
	}
}

//在粒子层面处理fluid, air, airsolo粒子与solid的碰撞关系，保证不会穿过边界到solid的内部。
__global__ void CollisionWithSolid_k(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray phisolid, farray sux, farray suy, farray suz, SCENE scene, float bounceVelParam, float bouncePosParam)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPESOLID)
			return;
		float3 ipos = ppos[idx];
		float3 ivel = pvel[idx];
		float iphi = getScaleFromFrid(ipos, phisolid);
		if (iphi <= 0.5f)		//靠近固体，距离只有半个格子
		{
			float3 svel = getParticleVelFromGrid(ipos, sux, suy, suz);
			float3 rvel = ivel - svel;
			float d = dparam.cellsize.x * 0.5f;
			float3 phigrad;
			phigrad.x = getScaleFromFrid(ipos + make_float3(d, 0, 0), phisolid) - getScaleFromFrid(ipos - make_float3(d, 0, 0), phisolid);
			phigrad.y = getScaleFromFrid(ipos + make_float3(0, d, 0), phisolid) - getScaleFromFrid(ipos - make_float3(0, d, 0), phisolid);
			phigrad.z = getScaleFromFrid(ipos + make_float3(0, 0, d), phisolid) - getScaleFromFrid(ipos - make_float3(0, 0, d), phisolid);
			if (length(phigrad) > 0)
			{
				phigrad = normalize(phigrad);		//指向外侧
				if (dot(rvel, phigrad)<0 || scene == SCENE_FREEZING)	//相对速度指向内侧
				{
					ivel -= bounceVelParam * dot(rvel, phigrad)*phigrad;		//法向速度置为与固体一样
					if (scene == SCENE_FREEZING)
						ivel -= 0.1f* (rvel - dot(rvel, phigrad)*phigrad);		//切向速度
				}
				ipos += bouncePosParam * phigrad * (0.5f - iphi) * dparam.cellsize.x;
			}
		}
		//并根据新的速度更新位置
		ipos += ivel*dparam.dt;
		//边界
		float rate = 0.5f, ratevel = -0.5f;
		if (pflag[idx] == TYPEAIRSOLO)
			rate = 0.8f, ratevel = -0.5f;
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(rate*dparam.cellsize.x));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(rate*dparam.cellsize.x));
		// 		if( ipos.x>tmax.x )
		// 			ivel.x *=ratevel, ipos.x=tmax.x;
		// 		if( ipos.x<tmin.x )
		// 			ivel.x *= ratevel, ipos.x=tmin.x;
		// 		if( ipos.y>tmax.y )
		// 			ivel.y *=ratevel, ipos.y=tmax.y;
		// 		if( ipos.y<tmin.y )
		// 			ivel.y *= ratevel, ipos.y=tmin.y;
		// 		if( ipos.z>tmax.z )
		// 			ivel.z *=ratevel, ipos.z=tmax.z;
		// 		if( ipos.z<tmin.z )
		// 			ivel.z *= ratevel, ipos.z=tmin.z;
		if (ipos.x <= tmin.x)
			ipos.x = tmin.x, ivel.x = 0.0f;
		if (ipos.y <= tmin.y)
			ipos.y = tmin.y, ivel.y = 0.0f;
		if (ipos.z <= tmin.z)
			ipos.z = tmin.z, ivel.z = 0.0f;

		if (ipos.x >= tmax.x)
			ipos.x = tmax.x, ivel.x = 0.0f;
		if (ipos.y >= tmax.y)
			ipos.y = tmax.y, ivel.y = 0.0f;
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//存储新的速度和位置
		pvel[idx] = ivel;
		ppos[idx] = ipos;
	}
}

//专门为melting and freezing场景写的，粒度要更细一些。在粒子层面处理fluid, air, airsolo粒子与solid的碰撞关系，保证不会穿过边界到solid的内部。防止穿透 penetration
__global__ void CollisionWithSolid_Freezing(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray phisolid, uint* gridstart, uint* gridend)
{
	
}

__global__ void buoyancyForSolid(float3 *ppos, float3 *pvel, char *pflag, int pnum, uint *gridstart, uint *gridend, float SolidBuoyanceParam)
{
	
}

__global__ void solidCollisionWithBound(float3 *ppos, float3 *pvel, char *pflag, int pnum, float SolidbounceParam, int nSolPoint)
{
	
}



//这个函数是考虑latent heat的主函数，当温度超过界限时(如固体的温度高于熔点)，则多余的热量放到latent heat里；当latent heat满足一定条件时，发生phase change.
__global__ void updateLatentHeat_k(float *parTemperature, float *parLHeat, char *partype, int pnum, float meltingpoint, float boilingpoint, float LiquidHeatTh)
{
	
}

__global__ void pouringwater(float3* pos, float3* vel, float* parmass, char* parflag, float *ptemperature, float *pLHeat, float *pGasContain, int parnum,
	float3 *ppourpos, float3 *ppourvel, char pourflag, int pournum, float *randfloat, int randnum, int frame, float posrandparam, float velrandparam,
	float defaultLiquidT, float LiquidHeatTh)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pournum)
	{
		//速度与位置的随机化
		int randbase = (frame + idx) % (randnum - 6);
		float3 randvel = make_float3(randfloat[randbase], randfloat[randbase + 1], randfloat[randbase + 2]) *2.0f - 1.0f;
		randbase += 3;
		float3 randpos = make_float3(randfloat[randbase], randfloat[randbase + 1], randfloat[randbase + 2]) *2.0f - 1.0f;

		pos[parnum + idx] = ppourpos[idx] + randpos * posrandparam * dparam.samplespace;
		vel[parnum + idx] = ppourvel[idx] + randvel * velrandparam;
		parmass[parnum + idx] = dparam.m0;
		parflag[parnum + idx] = pourflag;
		ptemperature[parnum + idx] = defaultLiquidT;
		pLHeat[parnum + idx] = LiquidHeatTh;
		pGasContain[parnum + idx] = 0;
	}
}


inline __device__ float getlen(float x, float y)
{
	return sqrt(x*x + y*y);
}
__global__ void initheat_grid_k(farray tp, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float x = i, z = k;
		float r = NX*0.15;

		if (getlen(x - NX / 4, z - NZ / 4) <= r)
			tp[idx] = 100, mark[idx] = TYPESOLID;
		else if (getlen(x - NX / 4 * 3, z - NZ / 4 * 3) <= r)
			tp[idx] = 0, mark[idx] = TYPEFLUID;
		else if (z<NZ / 2)
			tp[idx] = 20, mark[idx] = TYPEVACUUM;
		else
			tp[idx] = 80, mark[idx] = TYPEAIR;
	}
}
__global__ void set_softparticle_position(float3* solidParPos, float3* mParPos, float3* solidParVelFLIP,float3* mParVel, char* partype)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gnum)
	if (partype[idx]==TYPESOLID)
	{
		mParPos[idx] = solidParPos[idx];
		mParVel[idx] = (solidParVelFLIP[idx]+mParVel[idx])/2.0;
	//	mParVel[idx] = solidParVelFLIP[idx];
		
	}
};
