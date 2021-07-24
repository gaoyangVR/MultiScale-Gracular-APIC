#include <stdio.h>
#include<cuda_runtime.h>
#include<helper_math.h>
#include "timer.h"
#include "spray.h"

//record the images.
#include "SOIL.h"

extern int winw, winh;

matarray::matarray()
{
	data = NULL; xn = NX; yn = NY; zn = NZ;
}

f3array::f3array()
{
	data = NULL; xn = NX; yn = NY; zn = NZ;
}

farray::farray()
{
	data = NULL; xn = NX; yn = NY; zn = NZ;
}

charray::charray()
{
	data = NULL; xn = NX; yn = NY; zn = NZ;
}

void cspray::initparam()
{


	mscene = SCENE_SANDSTORM;

	stiffness = 0.8;  //   stiffness!
	SolidbounceParam = 01.50f;	
	mpause = false;
	boutputpovray = false;					
	boutputobj = false;				
	boutputfile = false;
	bOutputColoredParticle = false;
	outputframeDelta = 1;
	surfacetensionsigma = 0.0f;		
	heatalphafluid = 0.02f, heatalphaair = 0.008f;
	heatalphasolid = 0.2f, heatalphavacuum = 0.0001f;
	defaulttemperature = 293.15f, Temperature0 = 273.15f;
	meltingpoint = 273.15f, boilingpoint = 373.15f;
	pourNum = 0;
	m_bPouring = false;

	m_bAPIC = false;///////////////

	m_bSmoke = false;//////////////
	m_bSand = true;
	m_bSolid = false;
	m_bMelt = false;
	m_bFreeze = false;
	m_bFixSolid = true;
	m_bGenGas = false;
	m_bHeatTrans = true;
	m_bAddHeatBottom = false;
	m_bExtendHeatToBoundary = false;
	m_bCorrectPosition = true;
	mRecordImage = false;
	
	SolidBuoyanceParam = 0.;	
	defaultSolidT = 263.15f, defaultLiquidT = 293.15f, LiquidHeatTh = 10;
	posrandparam = 0.1f, velrandparam = 0.0f;
	alphaTempTrans = 0.0f;
	bounceVelParam = 1.0f, bouncePosParam = 0;
	seednum = 4;
	initdissolvegasrate = 1.0f, initgasrate = 0.7f;	//initial dissolved gas in liquid particle.
	vaporGenRate = 0.01f;
	temperatureMax_render = 373, temperatureMin_render = 273.0f;
	m_bSmoothMC = true;
	bRunMCSolid = true;
	m_DistanceFuncMC = 1;		
	
	frameMax = 700;
	m_bLiquidAndGas = false;
	m_bGas = false;				
	bubbleMaxVel = 6.5f;	
	m_bCPURun = false;
	updateSeedFrameDelta = 15;
	m_bCorrectFluidByAirLS = false;
	buoyanceRateAir = 1.25f, buoyanceRateSolo = 1.05f;
	m_beginFrame = 0;
	cntAirParMax = 0, cntLiquidMax = 0, cntSolidMax = 0, cntSandMax = 0, cntSmokeMax = 0;
	m_bBottomParticel = false;

	hparam.gravity = make_float3(0, 0, -9.8f);
	hparam.cellsize = make_float3(1.0f / 64);		
	hparam.samplespace = 0.5f*hparam.cellsize.x;

	//sprintf(outputdir, "output\\povray\\");
	//params relate to scene
	//APIC 
	
if (mscene == SCENE_SANDSTORM)
	{
		if (NX != 128 || NY != 128 || NZ != 128)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 1;
		initfluidparticle = 50000;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 10 + parNumNow*splitnum;	//pouring + init
		pourNum = 100;
		pourRadius = 0.5f;
		//pourRadius = 0.0f;
		//pourpos = make_float3(0.8f, 0.44f, 0.55f);
		//pourvel = make_float3(-01.5f, 0.0f, -1.f);
		pourpos = make_float3(0.5f, 0.44f, 0.55f);
		pourvel = make_float3(0.f, 0.f, 0.f);
		CompPouringParam_Freezing();	//水龙头
		if (boutputpovray)
			sprintf(outputdir, "output\\povraysandstorm\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objsandstorm\\");
		if (boutputfile)
			sprintf(outputdir, "outputfile\\in_sandstorm\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.6f, 0.5f, -.532f);				//固体位置
		wind = make_float3(3.0f, 10.0f, 15.0f); 
	}

	initHeatAlphaArray();

	rendermode = RENDER_PARTICLE;
	colormode = COLOR_PRESS;
	velmode = FLIP;
	if (m_bAPIC)
		velmode = APIC;

	bsmoothMC = true;
	mframe = 0;
	bColorRadius = false;
	bColorVel = false;

	m_btimer = true;

	bCouplingSphere = false;
	//default: for coupling with solid, simple code.
	spherepos = make_float3(0.5f, 0.5f, 0.5f);
	sphereradius = 0.1f;
	bmovesphere = false;
	spherevel = make_float3(-1.0f, 0.0f, 0.0f);

	//default parameters.
	if (mscene != SCENE_SANDSTORM)
		wind = make_float3(0.0f);
	hparam.centripetal = wind.y;
	hparam.tangential = wind.x;
	//param
	hparam.sin_phi = 0.3;
	hparam.miu = 0.5;
	hparam.cohesion = 0.0005;//

	hparam.gnnum = (NX + 1)*(NY + 1)*(NZ + 1);
	hparam.gnum = NX*NY*NZ;
	hparam.gvnum = make_int3((NX + 1)*NY*NZ, NX*(NY + 1)*NZ, NX*NY*(NZ + 1));
	hparam.dt = 0.0025f;		//
	if (m_bSmoke)
		hparam.dt = 0.0015f;//0.0015f  Pouring:0.0025
	hparam.gmin = make_float3(0.0f);
	hparam.gmax = hparam.cellsize.x * make_float3((float)NX, (float)NY, (float)NZ);		//x的长度总是1
	hparam.sandrho = 1600.0f;
	hparam.waterrho = 1000.0f;
	hparam.solidrho = 800.0f;
	hparam.m0 = (1.0f*hparam.waterrho) / pow(1.0f / hparam.samplespace, 3.0f);
	if (m_bSand)
		hparam.m0 = (1.0f*hparam.sandrho) / pow(1.0f / hparam.samplespace, 3.0f);
	//printf("\n%f\n\n", hparam.m0);
	hparam.airm0 = hparam.m0;
	//hparam.pradius = (float)(pow(hparam.m0 / hparam.waterrho*3.0f / 4.0f / M_PI, 1.0 / 3.0f));
	hparam.pradius = (float)(hparam.samplespace * pow(3.0f / 4.0f / M_PI, 1.0 / 3.0f));
	hparam.poly6kern = 315.0f / (64.0f * 3.141592f * pow(hparam.cellsize.x, 9.0f));
	hparam.spikykern = -45.0f / (3.141592f * pow(hparam.cellsize.x, 6.0f));
	hparam.lapkern = 45.0f / (3.141592f * pow(hparam.cellsize.x, 6.0f));

	maxVelForBubble = hparam.cellsize.x / hparam.dt;

	//MC
	NXMC = NX, NYMC = NY, NZMC = NZ;

	//time and parameters
	timeaver = new float[TIME_COUNT];
	timeframe = new float[TIME_COUNT];
	timemax = new float[TIME_COUNT];
	memset(timeaver, 0, sizeof(float)*TIME_COUNT);
	memset(timeframe, 0, sizeof(float)*TIME_COUNT);
	memset(timemax, 0, sizeof(float)*TIME_COUNT);

	//set cuda blocknum and threadnum.
	gsblocknum = (int)ceil(((float)NX*NY*NZ) / threadnum);
	int vnum = max((NX + 1)*NY*NZ, (NX)*(NY + 1)*NZ);
	vnum = max(vnum, NX*NY*(NZ + 1));
	gvblocknum = (int)ceil(((float)vnum) / threadnum);
	pblocknum = max(1, (int)ceil(((float)parNumNow) / threadnum));

	//MC parameters
	fMCDensity = 0.5f;
	smoothIterTimes = 5;

	velocitydissipation = 1.0f;//密度场空气阻力影响 0.995 1.0
	densedissipation = 1.0f;
	fDenseDiffuse = 0.001f;
	fVelocityDiffuse = 0.001f;
	nDiffuseIters = 4;
	correctionspring = 500000.0;
	correctionradius = 0.5;
	
	//sand to smoke threshold
	genSprayRelVelThres = 0.1f;//0.07 0.2 0.5
	genSprayVelThres = 1.0f;//0.7 1.0 1.3 
	
	//smoke to sand threshold
	genSandVelThres = 0.02f; //
	notgenSandvelThres = 10.f; //
	fWater2Spray = 1.0f;// 1.0 2.0 4.5

	//rand number
	randfloatcnt = 10000;
	renderpartiletype = TYPEAIR;

}

void cspray::init()
{	readdata_solid();	//
	initparam();
	initmem_bubble();
	printf("initmem complete.\n");
	initEmptyBubbles();		//

	initParticleGLBuffers();
	printf("initParticleGLBuffers complete.\n");

	initGridGLBuffers();
	printf("initGridGLBuffers complete.\n");
	initcubeGLBuffers();
	initdensityGLBuffers();
	initparticle_solidCoupling();
	printf("initparticle complete.\n");

	initdense();
	printf("initdense complete.\n");

	initlight();
	loadshader();
	printf("loadshader complete.\n");

	initMC();
	printf("initMC complete.\n");

	initTemperature();
	initSolubility();
	initSeedCell();
	if (mscene == SCENE_HEATTRANSFER)
		initheat_grid();

}

void cspray::rollrendermode()
{
	rendermode = (ERENDERMODE)((rendermode + 1) % RENDER_CNT);
	printf("rendermode=%d\n", (int)rendermode);
}
void cspray::rollcolormode(int delta)
{
	colormode = (ECOLORMODE)((colormode + delta + COLOR_CNT) % COLOR_CNT);
	printf("colormode=%d\n", (int)colormode);
}

void printTime(bool btime, char* info, CTimer &time)
{
	if (!btime)
		return;
	double deltatime = time.stopgetstartMS();
	printf("%lf--", deltatime);
	printf(info);
	printf("\n");
}

void printTime(bool btime, char* info, CTimer &time, float* ptimeaver, float* ptimeframe, int timeflag)
{
	if (!btime)
		return;
	double deltatime = time.stopgetstartMS();

	//statistics
	ptimeaver[timeflag] += (float)deltatime;
	ptimeframe[timeflag] += (float)deltatime;
}

void cspray::simstep()
{
 if (simmode == SIMULATION_APICBASIC)
			APIC_basicsim();

	//截图做视频
	static int frame = 0;
	if (!mpause && mRecordImage)
	{
		char str[100];
		//if( frame%2==0 )
		{
			sprintf(str, "output\\outputForHeatTransfer\\%05d.bmp", frame);
		//	SOIL_save_screenshot(str, SOIL_SAVE_TYPE_BMP, 0, 0, winw, winh);
		}
		frame++;
	}

	//统计时间
	if (mframe >= frameMax)
	{
		averagetime();
		exit(0);
	}
}
//water solidcoupling bubble
//apic mainbody
void cspray::APIC_basicsim()
{
}

void cspray::APIC_watersolidsim()
{
	
}

void cspray::watersim()
{
	
}

void cspray::waterSolidSim()
{
	
}

void cspray::smokesim()
{
	
}


void cspray::bubblesim()////////////////////////////mainbody
{
	if (!mpause)
	{
		//mpause=true;
		printf("Before MC: "), PrintMemInfo();
		
		CTimer time;
		CTimer time2;
		time2.startTimer();
		static CTimer timetotal;
		printTime(m_btimer, "TOTAL TIME!!", timetotal);

		memset(timeframe, 0, sizeof(float)*TIME_COUNT);

		if (m_btimer)
			printf("\n------------Frame %d:-------------\n", mframe);
	
		if (mframe%updateSeedFrameDelta == 0)
			updateSeedCell();

		time.startTimer();

		if (mscene == SCENE_SANDSTORM)
			setwind();

		//APIC: basic simulation
		{
			//SANDSTORM
			if (mscene == SCENE_SANDSTORM && mframe == 1300)
			{
				wind = make_float3(0.0f);
				//printf("wind => 0");
			}
			//1. external force: gravity, buoyancy, surface tension
			hashAndSortParticles();
			addexternalforce();
			//printTime( m_btimer, "addexternalforce", time);
			printTime(m_btimer, "addexternalforce", time2);

			
			//sandmark & mmark initialize here
			computeLevelset(0);	//todo: emptybubble
			sweepLSAndMardGrid();
			printTime(m_btimer, "computeLevelset", time2);

						markgrid();// = mark_boundary_cell
			printTime(m_btimer, "markgrid_bubble", time2);

			//3. grid-based solver

			mapvelp2g_bubble();
			printTime(m_btimer, "mapvelp2g", time2);

			sweepPhi(phifluid, TYPEFLUID);
			sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);
			setWaterBoundaryU(waterux, wateruy, wateruz);
			printTime(m_btimer, "sweepU", time2);

			//空气和液体有两个速度场，统一计算压强并更新			//Section 3.2

			project_CG_bubble();
			printTime(m_btimer, "project_CG_bubble", time2);

			sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);
			setWaterBoundaryU(waterux, wateruy, wateruz);

			printTime(m_btimer, "sweepU", time2);

			mapvelg2p_bubble();
			printTime(m_btimer, "mapvelg2p_bubble", time2);

			//4. solid simulation step.
			
			if (m_bFixSolid)
				solidmotion_fixed();
			
			solidmotion();
			advect_bubble();
			//printTime( m_btimer, "advect", time);
			printTime(m_btimer, "advect_bubble", time2);
			
			hashAndSortParticles();
			//******particle handling
			CollisionSolid();
			printTime(m_btimer, "CollisionSolid", time2);

			//6. correct all positions, distribute evenly.
			if (m_bCorrectPosition)
			{
				hashAndSortParticles();
				correctpos();			//todo: emptybubble
								
				hashAndSortParticles();
				computeLevelset(0.15f);
				sweepLSAndMardGrid();
			}
			else
			{
				hashAndSortParticles();
				computeLevelset(0.15f);
				sweepLSAndMardGrid();
			}
			printTime(m_btimer, "dynamics", time, timeaver, timeframe, TIME_DYNAMICS);
			printTime(m_btimer, "correctpos", time2);

			//solid model output
			if (m_bSolid)
			{
				if (mframe%outputframeDelta == 0)
				{
					preMC();
					runMC_solid();				
				}
				printTime(m_btimer, "runMC_Solid", time2);
			}
			runMC_fluid();
		}

		//transfer: from sand to smoke
		if (m_bSmoke)
		{
			hashAndSortParticles();
			generatespray();
			printTime(m_btimer, "generatespray", time2);
		}
		

		//spray density simulation.
		if (m_bSmoke)
		{
			//smokesetvel();
			updatesprayvel();
			printTime(m_btimer, "updatesprayvel", time2);

			smokemarkgrid();
			printTime(m_btimer, "smokemarkgrid", time2);

			setSmokeBoundaryU(msprayux, msprayuy, msprayuz);
			printTime(m_btimer, "setBoundaryU", time2);

			project_Jacobi(msprayux, msprayuy, msprayuz);   
			printTime(m_btimer, "project", time2);

			setSmokeBoundaryU(msprayux, msprayuy, msprayuz);
			printTime(m_btimer, "setBoundaryU", time2);

			smokeadvection();
			printTime(m_btimer, "smokeadvection", time2);
		}

		if (m_bSmoke)
		{
			hashAndSortParticles();
			generatesand();
			printTime(m_btimer, "genaratesand", time2);
		}
		{
			//for render!!!
			hashAndSortParticles();

			if (boutputfile && mframe%outputframeDelta == 0)
			{				
				outputBPhysFile(mframe / outputframeDelta, mParPos, parmass, parflag, parNumNow);
			}

			printTime(m_btimer, "outpout", time2);

			copyDensity2GL();
			printTime(m_btimer, "copyDensity2GL", time2);
			copyParticle2GL();
			printTime(m_btimer, "copyParticle2GL", time2);
		}

		printTime(m_btimer, "dynamics", time, timeaver, timeframe, TIME_DISPLAY);
		printTime(m_btimer, "MC", time2);
		//time statistics;
		for (int i = 0; i < TIME_COUNT; i++)
		{
			if (timemax[i] < timeframe[i])
				timemax[i] = timeframe[i];
			printf("timemax %d: %f\n", i, timeframe[i]);
		}

		if (mframe % 5 == 0)
			statisticParticleflag(mframe, parflag, parNumNow);

		mframe++;
	}
}

void cspray::resetsim()
{
	parNumNow = 0;
	int gsmemsize = sizeof(float)*hparam.gnum;
	cudaMemset(mpress.data, 0, gsmemsize);
	cudaMemset(temppress.data, 0, gsmemsize);
	cudaMemset(mDiv.data, 0, gsmemsize);
	cudaMemset(phifluid.data, 0, gsmemsize);
	cudaMemset(phiair.data, 0, gsmemsize);

	//u
	int gvxmemsize = sizeof(float)*hparam.gvnum.x;
	int gvymemsize = sizeof(float)*hparam.gvnum.y;
	int gvzmemsize = sizeof(float)*hparam.gvnum.z;
	int gvnmemsize = sizeof(float3)*hparam.gnnum;

	cudaMemset(waterux.data, 0, gvxmemsize);
	cudaMemset(wateruy.data, 0, gvymemsize);
	cudaMemset(wateruz.data, 0, gvzmemsize);
	cudaMemset(waterux_old.data, 0, gvxmemsize);
	cudaMemset(wateruy_old.data, 0, gvymemsize);
	cudaMemset(wateruz_old.data, 0, gvzmemsize);
	cudaMemset(solidvel.data, 0, gvnmemsize);

	cudaMemset(mParVel, 0, parNumMax*sizeof(float3));

	cudaMemset(spraydense.data, 0, sizeof(float)*hparam.gnum);
	cudaMemset(msprayux.data, 0, gvxmemsize);
	cudaMemset(msprayuy.data, 0, gvymemsize);
	cudaMemset(msprayuz.data, 0, gvzmemsize);
}

void cspray::averagetime()
{
	for (int i = 0; i < TIME_COUNT; i++)
		timeaver[i] /= mframe;
	//输出
	char str[100];
	sprintf(str, "statistictime%d.txt", (int)(mscene));
	FILE* fp = fopen(str, "w");
	fprintf(fp, "max liquid particle=%d, max gas particle=%d, max solid particle=%d\n\n", cntLiquidMax, cntAirParMax, cntSolidMax);
	fprintf(fp, "time:\n");
	for (int i = 0; i < TIME_COUNT; i++)
	{
		fprintf(fp, "time_index=%d, timeaver=%.2f, timemax=%.2f\n", i, timeaver[i], timemax[i]);
	}
	fclose(fp);
}

void cspray::movesphere()
{
	if (!bmovesphere)
		return;
	solidInitPos += spherevel * hparam.dt;
	if (solidInitPos.x < 0.5f)
	{
		solidInitPos.x = 0.5f;
		spherevel.x = abs((long)spherevel.x);
	}
	if (solidInitPos.x>0.9f)
	{
		solidInitPos.x = 0.9f;
		spherevel.x = -abs((long)spherevel.x);
	}
}

float3 cspray::mapColorBlue2Red_h(float v)
{
	float3 color;
	if (v < 0)
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

void cspray::statisticParticleflag(int frame, char *dflag, int pnum)
{
	static char *hflag = new char[parNumMax];
	cudaMemcpy(hflag, dflag, pnum*sizeof(char), cudaMemcpyDeviceToHost);
	int cntAirPar = 0, cntLiquid = 0, cntSolid = 0, cntSand = 0, cntSmoke = 0;
	for (int i = 0; i < pnum; i++)
	{
		if (hflag[i] == TYPEAIRSOLO || hflag[i] == TYPEAIR)
			cntAirPar++;
		else if (hflag[i] == TYPEFLUID)
			cntLiquid++;
		else if (hflag[i] == TYPESOLID)
			cntSolid++;
		//else if (hflag[i] == TYPESAND)
		//	cntSand++;
		else if (hflag[i] == TYPESMOKE)
			cntSmoke++;
	}
	cntLiquidMax = max(cntLiquid, cntLiquidMax);
	cntAirParMax = max(cntAirPar, cntAirParMax);
	cntSolidMax = max(cntSolid, cntSolidMax);
	cntSandMax = max(cntSand, cntSandMax);
	cntSmokeMax = max(cntSmoke, cntSmokeMax);

	//printf( "droplet particle: %d, liquid particle: %d\n", cntAirPar, cntLiquid );
}

class MyMatrix33
{
public:
	MyMatrix33(
		float a00, float a01, float a02,
		float a10, float a11, float a12,
		float a20, float a21, float a22)
	{
		data[0][0] = a00;
		data[0][1] = a01;
		data[0][2] = a02;
		data[1][0] = a10;
		data[1][1] = a11;
		data[1][2] = a12;
		data[2][0] = a20;
		data[2][1] = a21;
		data[2][2] = a22;
	}

	float3 Multiple(const float3 &v)
	{
		float3 result;
		result.x = data[0][0] * v.x + data[1][0] * v.y + data[2][0] * v.z;
		result.y = data[0][1] * v.x + data[1][1] * v.y + data[2][1] * v.z;
		result.z = data[0][2] * v.x + data[1][2] * v.y + data[2][2] * v.z;
		return result;
	}

private:
	float data[3][3];
};
void cspray::CompPouringParam_Break()
{
	//if (mframe==0)
	{
		//
	
		float3 *hpourpos = new float3[initfluidparticle];
		float3 *hpourvel = new float3[initfluidparticle];
	
		int cnt = 0;
		float3 ipos; int i = 0;
	
		for (float z = hparam.cellsize.x + hparam.samplespace; z < 0.8f * NZ*hparam.cellsize.x && i < initfluidparticle; z += hparam.samplespace)
		{
			for (float y = hparam.cellsize.x + hparam.samplespace; y < hparam.cellsize.x*(NY - 1) - 0.5f*hparam.samplespace && i < initfluidparticle; y += hparam.samplespace)
			for (float x = hparam.cellsize.x + hparam.samplespace; x < hparam.cellsize.x*(NX - 1)*0.5 - hparam.samplespace && i < initfluidparticle; x += hparam.samplespace)
			{
				hpourpos[i] = make_float3(x, y, z);
				hpourvel[i] = make_float3(0, 0, 0);
				
				++i;
			}
		}
			printf("pouring num=%d\n", i);

			cudaMalloc((void**)&dpourpos, sizeof(float3)*initfluidparticle);
			cudaMalloc((void**)&dpourvel, sizeof(float3)*initfluidparticle);
		
			cudaMemcpy(dpourpos, hpourpos, sizeof(float3)*initfluidparticle, cudaMemcpyHostToDevice);
			cudaMemcpy(dpourvel, hpourvel, sizeof(float3)*initfluidparticle, cudaMemcpyHostToDevice);
			
		
		delete[] hpourpos;
		delete[] hpourvel; 
	}
}
void cspray::CompPouringParam_Freezing()
{
	//
	int num = (int)(pourRadius / hparam.samplespace)*3;

	int memnum = (2 * num + 1)*(2 * num + 1) ;
	float3 *hpourpos = new float3[memnum];
	float3 *hpourvel = new float3[memnum];
	int cnt = 0;
	float3 ipos;
	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos + make_float3(x*hparam.samplespace, y*hparam.samplespace, 0);
		if (length(ipos - pourpos) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}
	if (cnt == 0)
		printf("pouring num=0!!!\n");
	else
	{
		printf("pouring num=%d\n", cnt);
		
		cudaMalloc((void**)&dpourpos, sizeof(float3)*cnt);
		cudaMalloc((void**)&dpourvel, sizeof(float3)*cnt);
		cudaMemcpy(dpourpos, hpourpos, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(dpourvel, hpourvel, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
	}
	pourNum = cnt;
	
	delete[] hpourpos;
	delete[] hpourvel;
}

void cspray::CompPouringParam_Ineraction()
{
	//
	float ss = hparam.samplespace * 2.0f;
	int num = (int)(pourRadius / ss);
	int memnum = (2 * num + 1)*(2 * num + 1) * 2;
	float3 *hpourpos = new float3[memnum];
	float3 *hpourvel = new float3[memnum];
	int cnt = 0;
	float3 ipos;
	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos + make_float3(x*ss, y*ss, 0);
		if (length(ipos - pourpos) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}
	if (cnt == 0)
		printf("pouring num=0!!!\n");
	else
	{
		printf("pouring num=%d\n", cnt);
		cudaMalloc((void**)&dpourpos, sizeof(float3)*cnt);
		cudaMalloc((void**)&dpourvel, sizeof(float3)*cnt);
		cudaMemcpy(dpourpos, hpourpos, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(dpourvel, hpourvel, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
	}
	pourNum = cnt;

	delete[] hpourpos;
	delete[] hpourvel;
}

void cspray::CompPouringParam_Ineraction2()
{
	//
	float ss = hparam.samplespace * 2.0f;
	int num = (int)(pourRadius / ss);
	int memnum = (2 * num + 1)*(2 * num + 1) * 4;
	float3 *hpourpos = new float3[memnum];
	float3 *hpourvel = new float3[memnum];
	int cnt = 0;
	float3 ipos;
	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos + make_float3(x*ss, y*ss, 0);
		if (length(ipos - pourpos) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}

	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos2 + make_float3(x*ss, y*ss, 0);
		if (length(ipos - pourpos2) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}

	if (cnt == 0)
		printf("pouring num=0!!!\n");
	else
	{
		printf("pouring num=%d\n", cnt);
		cudaMalloc((void**)&dpourpos, sizeof(float3)*cnt);
		cudaMalloc((void**)&dpourvel, sizeof(float3)*cnt);
		cudaMemcpy(dpourpos, hpourpos, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(dpourvel, hpourvel, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
	}
	pourNum = cnt;

	delete[] hpourpos;
	delete[] hpourvel;
}

