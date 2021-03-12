// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include <Windows.h>
#include <string>
#include "Shobjidl.h"

extern "C" {
	__declspec(dllexport) void SetWallpaper(int id, const wchar_t* s) {
		WCHAR* mid = new WCHAR[200];
		WCHAR* temp = mid;
		memset(mid, 0, 200 * 2);
		CoInitialize(NULL);

		IDesktopWallpaper* p;
		if (SUCCEEDED(CoCreateInstance(
			__uuidof(DesktopWallpaper),
			0,
			CLSCTX_LOCAL_SERVER,
			__uuidof(IDesktopWallpaper),
			(void**)&p
		))) {
			p->GetMonitorDevicePathAt(id, &mid);  //0: the first monitor  1: the second monitor
			p->SetWallpaper(mid, s);
			p->Release();
		}
		else {
			std::terminate();
		}
		delete[] temp;
		CoUninitialize();
	}
	__declspec(dllexport) void SetProgressState(HWND hwnd, TBPFLAG tbpFlags) {
		CoInitialize(NULL);

		ITaskbarList3* p;
		if (SUCCEEDED(CoCreateInstance(
			CLSID_TaskbarList,
			0,
			CLSCTX_INPROC_SERVER,
			IID_ITaskbarList3,
			(void**)&p
		))) {
			p->SetProgressState(hwnd, tbpFlags);
			p->Release();
		}
		else {
			std::terminate();
		}
		CoUninitialize();
	}
	__declspec(dllexport) void SetProgressValue(HWND hwnd, ULONGLONG ullCompleted, ULONGLONG ullTotal) {
		CoInitialize(NULL);

		ITaskbarList3* p;
		if (SUCCEEDED(CoCreateInstance(
			CLSID_TaskbarList,
			0,
			CLSCTX_INPROC_SERVER,
			IID_ITaskbarList3,
			(void**)&p
		))) {
			p->SetProgressValue(hwnd, ullCompleted, ullTotal);
			p->Release();
		}
		else {
			std::terminate();
		}
		CoUninitialize();
	}
}

BOOL APIENTRY DllMain(
	HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
) {
	switch (ul_reason_for_call) {
		case DLL_PROCESS_ATTACH:
		case DLL_THREAD_ATTACH:
		case DLL_THREAD_DETACH:
		case DLL_PROCESS_DETACH:
			break;
	}
	return TRUE;
}