#pragma warning(disable:26451)
/*++

Copyright (c) 1999-2002  Microsoft Corporation

Module Name:

    scanUser.c

Abstract:

    This file contains the implementation for the main function of the
    user application piece of scanner.  This function is responsible for
    actually scanning file contents.

Environment:

    User mode

--*/

#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <winioctl.h>
#include <string.h>
#include <crtdbg.h>
#include <assert.h>
#include <fltuser.h>
#include "scanuk.h"
#include "scanuser.h"
#include <dontuse.h>

//
//  Default and Maximum number of threads.
//

#define SCANNER_DEFAULT_REQUEST_COUNT       5
#define SCANNER_DEFAULT_THREAD_COUNT        2
#define SCANNER_MAX_THREAD_COUNT            64

UCHAR FoulString[] = "foul";

//
//  Context passed to worker threads
//

typedef struct _SCANNER_THREAD_CONTEXT {

    HANDLE Port;
    HANDLE Completion;

} SCANNER_THREAD_CONTEXT, *PSCANNER_THREAD_CONTEXT;


VOID
Usage (
    VOID
    )
/*++

Routine Description

    Prints usage

Arguments

    None

Return Value

    None

--*/
{

    printf( "Connects to the scanner filter and scans buffers \n" );
    printf( "Usage: scanuser [requests per thread] [number of threads(1-64)]\n" );
}

BOOLEAN
ReadFileDataW(
	PWCHAR pwPath,
	PUCHAR *ppBuffer,
	PULONG pSize
	)
{
	HANDLE hFile = INVALID_HANDLE_VALUE;
	PUCHAR pBuffer = NULL;
	DWORD dwFileSize = 0;
	DWORD dwHighFileSize = 0;
	ULONG BytesRead = 0;
	BOOLEAN bRet = FALSE;

	if(ppBuffer)
		*ppBuffer = NULL;

	hFile = CreateFileW(
				pwPath,
				GENERIC_READ,
				FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE, // Important
				NULL,
				OPEN_EXISTING,
				FILE_ATTRIBUTE_NORMAL,
				NULL
				);
	if(INVALID_HANDLE_VALUE == hFile)
	{
		printf("[ReadFileDataW] CreateFile Failed(LastError=0n%d) %ws\r\n", GetLastError(), pwPath);
		goto $RET;
	}

	dwFileSize = GetFileSize(hFile, &dwHighFileSize);
	if(INVALID_FILE_SIZE == dwFileSize)
	{
		printf("[ReadFileDataW] GetFileSize Failed(LastError=0n%d) %ws\r\n", GetLastError(), pwPath);
		goto $RET;
	}

	if(0 != dwHighFileSize)
	{
		printf("[ReadFileDataW] if(0 != dwHighFileSize) %ws\r\n", pwPath);
		goto $RET;
	}

	if(ppBuffer)
	{
		pBuffer = (PUCHAR)malloc(dwFileSize+2); 
		if(NULL == pBuffer)
		{
			printf("[ReadFileDataW] malloc Failed(size=%d) %ws\r\n", dwFileSize+2, pwPath);
			goto $RET;
		}
		
		RtlZeroMemory(pBuffer, dwFileSize+2);

		SetFilePointer(hFile, 0, 0, FILE_BEGIN);

		if(!ReadFile(hFile, pBuffer, dwFileSize, &BytesRead, NULL))
		{
			printf("[ReadFileDataW] ReadFile Failed(LastError=0n%d) %ws\r\n", GetLastError(), pwPath);
			goto $RET;
		}

		if(dwFileSize != BytesRead)
		{
			printf("[ReadFileDataW] dwFileSize != BytesRead %ws\r\n", pwPath);
			goto $RET;
		}

		*ppBuffer = pBuffer;

		if(pSize)
			*pSize = BytesRead;
	}

	bRet = TRUE;

$RET:

	if(!bRet)
		free(pBuffer);

	if(INVALID_HANDLE_VALUE != hFile)
		CloseHandle(hFile);

	return bRet;
}

BOOL
ScanBuffer (
    __in PWCHAR pwFilePath
    )
/*++

Routine Description

    Scans the supplied buffer for an instance of FoulString.

    Note: Pattern matching algorithm used here is just for illustration purposes,
    there are many better algorithms available for real world filters

Arguments

    Buffer      -   Pointer to buffer
    BufferSize  -   Size of passed in buffer

Return Value

    TRUE        -    Found an occurrence of the appropriate FoulString
    FALSE       -    Buffer is ok

--*/
{
    PUCHAR p;
	PUCHAR Buffer = NULL;
    ULONG BufferSize = 0;
	BOOL bRet = FALSE;
    ULONG searchStringLength = sizeof(FoulString) - sizeof(UCHAR);

	if(!ReadFileDataW(pwFilePath, &Buffer, &BufferSize))
		goto $RET;
	
    for (p = Buffer;
         p <= (Buffer + BufferSize - searchStringLength);
         p++) {

        if (RtlEqualMemory( p, FoulString, searchStringLength )) {

            printf( "Found a string\n" );

            //
            //  Once we find our search string, we're not interested in seeing
            //  whether it appears again.
            //

            bRet = TRUE;
			goto $RET;
        }
    }

$RET:

	if(Buffer)
		free(Buffer);

    return bRet;
}

PWCHAR
GetCharPointerW(
	PWCHAR pwStr,
	WCHAR wLetter,
	int Count
	)
{
	int i = 0, j = 0;
 //	int cchStr = (int)wcslen(pwStr);

	for(j = 0; j < Count; j++)
	{
		while(pwStr[i] != wLetter && pwStr[i] != 0)
		{
			++i;
		}
		++i;
	}

	return (PWCHAR)&pwStr[--i];
}

DWORD
ScannerWorker(
    __in PSCANNER_THREAD_CONTEXT Context
    )
/*++

Routine Description

    This is a worker thread that


Arguments

    Context  - This thread context has a pointer to the port handle we use to send/receive messages,
                and a completion port handle that was already associated with the comm. port by the caller

Return Value

    HRESULT indicating the status of thread exit.

--*/
{
    PSCANNER_NOTIFICATION notification;
    SCANNER_REPLY_MESSAGE replyMessage;
    PSCANNER_MESSAGE message;
    LPOVERLAPPED pOvlp;
    BOOL result;
    DWORD outSize;
    HRESULT hr;
    ULONG_PTR key;

#pragma warning(push)
#pragma warning(disable:4127) // conditional expression is constant

    while (TRUE) {

#pragma warning(pop)

        //
        //  Poll for messages from the filter component to scan.
        //

        result = GetQueuedCompletionStatus( Context->Completion, &outSize, &key, &pOvlp, INFINITE );

        //
        //  Obtain the message: note that the message we sent down via FltGetMessage() may NOT be
        //  the one dequeued off the completion queue: this is solely because there are multiple
        //  threads per single port handle. Any of the FilterGetMessage() issued messages can be
        //  completed in random order - and we will just dequeue a random one.
        //

        message = CONTAINING_RECORD( pOvlp, SCANNER_MESSAGE, Ovlp );

        if (!result) {

            //
            //  An error occured.
            //

            hr = HRESULT_FROM_WIN32( GetLastError() );
            break;
        }

        printf( "Received message, size %lld\n", pOvlp->InternalHigh );

        notification = &message->Notification;

        assert(notification->BytesToScan <= SCANNER_READ_BUFFER_SIZE);
        __analysis_assume(notification->BytesToScan <= SCANNER_READ_BUFFER_SIZE);

		{
			WCHAR wDosFilePath[512] = {0,};
			
			PWCHAR pwPtr = GetCharPointerW((PWCHAR)notification->Contents, L'\\', 3);
			if(!pwPtr)
			{
				hr = E_FAIL;
				break;
			}
			
			*pwPtr = L'\0';
			
			hr = FilterGetDosName((PWCHAR)notification->Contents, wDosFilePath, 512);
			if(FAILED(hr))
				break;
			
			wcscat(wDosFilePath, L"\\");
			wcscat(wDosFilePath, ++pwPtr);
			
			printf( "FilePath: %ws\n", wDosFilePath );
			
			result = ScanBuffer( wDosFilePath );
		}

        replyMessage.ReplyHeader.Status = 0;
        replyMessage.ReplyHeader.MessageId = message->MessageHeader.MessageId;

        //
        //  Need to invert the boolean -- result is true if found
        //  foul language, in which case SafeToOpen should be set to false.
        //

        replyMessage.Reply.SafeToOpen = !result;

        printf( "Replying message, SafeToOpen: %d\n", replyMessage.Reply.SafeToOpen );

        hr = FilterReplyMessage( Context->Port,
                                 (PFILTER_REPLY_HEADER) &replyMessage,
                                 sizeof( replyMessage ) );

        if (SUCCEEDED( hr )) {

            printf( "Replied message\n" );

        } else {

            printf( "Scanner: Error replying message. Error = 0x%X\n", hr );
            break;
        }

        memset( &message->Ovlp, 0, sizeof( OVERLAPPED ) );

        hr = FilterGetMessage( Context->Port,
                               &message->MessageHeader,
                               FIELD_OFFSET( SCANNER_MESSAGE, Ovlp ),
                               &message->Ovlp );

        if (hr != HRESULT_FROM_WIN32( ERROR_IO_PENDING )) {

            break;
        }
    }

    if (!SUCCEEDED( hr )) {

        if (hr == HRESULT_FROM_WIN32( ERROR_INVALID_HANDLE )) {

            //
            //  Scanner port disconncted.
            //

            printf( "Scanner: Port is disconnected, probably due to scanner filter unloading.\n" );

        } else {

            printf( "Scanner: Unknown error occured. Error = 0x%X\n", hr );
        }
    }

    free( message );

    return hr;
}


int _cdecl
main (
    __in int argc,
    __in_ecount(argc) char *argv[]
    )
{
    DWORD requestCount = SCANNER_DEFAULT_REQUEST_COUNT;
    DWORD threadCount = SCANNER_DEFAULT_THREAD_COUNT;
    HANDLE threads[SCANNER_MAX_THREAD_COUNT];
    SCANNER_THREAD_CONTEXT context;
    HANDLE port, completion;
    PSCANNER_MESSAGE msg;
    DWORD threadId;
    HRESULT hr;
    DWORD i, j;

    //
    //  Check how many threads and per thread requests are desired.
    //

    if (argc > 1) {

        requestCount = atoi( argv[1] );

        if (requestCount <= 0) {

            Usage();
            return 1;
        }

        if (argc > 2) {

            threadCount = atoi( argv[2] );
        }

        if (threadCount <= 0 || threadCount > 64) {

            Usage();
            return 1;
        }
    }

    //
    //  Open a commuication channel to the filter
    //

    printf( "Scanner: Connecting to the filter ...\n" );

    hr = FilterConnectCommunicationPort( ScannerPortName,
                                         0,
                                         NULL,
                                         0,
                                         NULL,
                                         &port );

    if (IS_ERROR( hr )) {

        printf( "ERROR: Connecting to filter port: 0x%08x\n", hr );
        return 2;
    }

    //
    //  Create a completion port to associate with this handle.
    //

    completion = CreateIoCompletionPort( port,
                                         NULL,
                                         0,
                                         threadCount );

    if (completion == NULL) {

        printf( "ERROR: Creating completion port: %d\n", GetLastError() );
        CloseHandle( port );
        return 3;
    }

    printf( "Scanner: Port = 0x%p Completion = 0x%p\n", port, completion );

    context.Port = port;
    context.Completion = completion;

    //
    //  Create specified number of threads.
    //

    for (i = 0; i < threadCount; i++) {

        threads[i] = CreateThread( NULL,
                                   0,
                                   ScannerWorker,
                                   &context,
                                   0,
                                   &threadId );

        if (threads[i] == NULL) {

            //
            //  Couldn't create thread.
            //

            hr = GetLastError();
            printf( "ERROR: Couldn't create thread: %d\n", hr );
            goto main_cleanup;
        }

        for (j = 0; j < requestCount; j++) {

            //
            //  Allocate the message.
            //

#pragma prefast(suppress:__WARNING_MEMORY_LEAK, "msg will not be leaked because it is freed in ScannerWorker")
            msg = malloc( sizeof( SCANNER_MESSAGE ) );

            if (msg == NULL) {

                hr = ERROR_NOT_ENOUGH_MEMORY;
                goto main_cleanup;
            }

            memset( &msg->Ovlp, 0, sizeof( OVERLAPPED ) );

            //
            //  Request messages from the filter driver.
            //

            hr = FilterGetMessage( port,
                                   &msg->MessageHeader,
                                   FIELD_OFFSET( SCANNER_MESSAGE, Ovlp ),
                                   &msg->Ovlp );

            if (hr != HRESULT_FROM_WIN32( ERROR_IO_PENDING )) {

                free( msg );
                goto main_cleanup;
            }
        }
    }

    hr = S_OK;

    WaitForMultipleObjectsEx( i, threads, TRUE, INFINITE, FALSE );

main_cleanup:

    printf( "Scanner:  All done. Result = 0x%08x\n", hr );

    CloseHandle( port );
    CloseHandle( completion );

    return hr;
}

