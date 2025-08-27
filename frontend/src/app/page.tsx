"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import Timeline from "@/components/timeline";
import {
  Loader2,
  Download,
  Trash2,
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  Square,
  Video,
} from "lucide-react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";

// 타입 정의
interface UploadedFile {
  id: string;
  fileName: string;
  originalName?: string;
  fileType: string;
  fileSize: number;
  uploadedAt: Date;
  previewUrl: string;
  status: "uploaded" | "processing" | "completed" | "error";
  processingResults?: ProcessingResult[];
  fileUrl?: string;
}

interface ProcessingResult {
  id: string;
  actionId: string;
  actionLabel: string;
  status: "completed";
  resultUrl: string;
  resultPreviewUrl: string;
  completedAt: number;
}

interface ProcessStep {
  id: string;
  action: string;
  description: string;
  resultUrl: string;
  timestamp: Date;
  completedAt: Date;
}

interface ActionButton {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface UploadProgressInfo {
  isUploading: boolean;
  progress: number;
  message: string;
}

export default function Home() {
  // 상태 관리
  const [currentMode, setCurrentMode] = useState<"image" | "video">("image");
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);
  const [processHistory, setProcessHistory] = useState<ProcessStep[]>([]);
  const [videoFrames, setVideoFrames] = useState<string[]>([]);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [uploadProgress, setUploadProgress] = useState<UploadProgressInfo>({
    isUploading: false,
    progress: 0,
    message: "",
  });

  // 비디오 편집 관련 상태
  const [timelineZoom, setTimelineZoom] = useState(1);
  const [selectedFrames, setSelectedFrames] = useState<Set<number>>(new Set());
  const [isPlaying, setIsPlaying] = useState(false);
  const [draggedFrame] = useState<number | null>(null);

  const [preloadedFrames, setPreloadedFrames] = useState<Set<number>>(
    new Set()
  );
  const [videoFps, setVideoFps] = useState(30); // 비디오 FPS
  const [videoDuration, setVideoDuration] = useState(0); // 비디오 총 길이(초)

  // 가상 스크롤링 및 성능 최적화 함수들
  // 사용하지 않는 함수들 제거됨

  // 프레임 프리로딩 관리 (안정화된 버전)
  const preloadFrame = useCallback(
    (index: number) => {
      if (videoFrames[index]) {
        const img = new Image();
        img.onload = () => {
          setPreloadedFrames((prev) => {
            if (prev.has(index)) return prev; // 이미 로드된 경우 상태 변경 방지
            return new Set(prev).add(index);
          });
        };
        img.src = videoFrames[index];
      }
    },
    [videoFrames]
  );

  // 비디오 플레이어 최적화
  const optimizedPlayFrame = (frameIndex: number) => {
    // 현재 프레임과 다음 몇 프레임을 프리로드
    for (
      let i = frameIndex;
      i < Math.min(frameIndex + 3, videoFrames.length);
      i++
    ) {
      preloadFrame(i);
    }
    setSelectedFrameIndex(frameIndex);
  };

  // 비디오 재생/일시정지
  const togglePlayPause = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // 비디오 재생 효과 (별도 useEffect로 관리)
  useEffect(() => {
    let playInterval: NodeJS.Timeout | null = null;

    if (isPlaying && videoFrames.length > 0) {
      playInterval = setInterval(() => {
        setSelectedFrameIndex((prev) => {
          const nextFrame = prev + 1;
          if (nextFrame >= videoFrames.length) {
            setIsPlaying(false);
            return prev; // 마지막 프레임에서 정지
          }
          return nextFrame;
        });
      }, 1000 / videoFps); // 원본 비디오 FPS로 재생
    }

    return () => {
      if (playInterval) {
        clearInterval(playInterval);
      }
    };
  }, [isPlaying, videoFrames.length, videoFps]);

  // 액션 버튼 정의
  const imageActions: ActionButton[] = [
    {
      id: "remove-background",
      label: "배경제거",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 12H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
    },
    {
      id: "upscale-x2",
      label: "확대x2",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10h-3m1.5-1.5v3"
          />
        </svg>
      ),
    },
    {
      id: "upscale-x4",
      label: "확대x4",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM16 10h-6m3-3v6"
          />
        </svg>
      ),
    },
    {
      id: "vectorize-color",
      label: "벡터화",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5H9"
          />
        </svg>
      ),
    },
    {
      id: "vectorize-bw",
      label: "흑백벡터",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10"
          />
        </svg>
      ),
    },
    {
      id: "style-retro",
      label: "레트로",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m0 0V1h2a2 2 0 012 2v18a2 2 0 01-2 2H5a2 2 0 01-2-2V3a2 2 0 012-2h2v3"
          />
        </svg>
      ),
    },
    {
      id: "style-painting",
      label: "페인팅",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
          />
        </svg>
      ),
    },
  ];

  const videoActions: ActionButton[] = [
    {
      id: "extract-last-frame",
      label: "마지막프레임",
      icon: ({ className }) => (
        <svg
          className={className}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
          />
        </svg>
      ),
    },
  ];

  const currentActions = currentMode === "image" ? imageActions : videoActions;

  // 백엔드에서 파일 목록 로드
  const loadFilesFromBackend = async () => {
    try {
      const response = await fetch("http://localhost:9090/api/files");
      if (response.ok) {
        const data = await response.json();
        const files: UploadedFile[] = data.files.map(
          (file: Record<string, unknown>) => ({
            id: file.id as string,
            fileName: (file.name || file.fileName) as string,
            originalName: (file.name || file.fileName) as string,
            fileType: file.fileType as string,
            fileSize: (file.size || file.fileSize || 0) as number,
            uploadedAt: new Date(file.uploadedAt as string | number | Date),
            previewUrl: (file.preview_url || file.previewUrl || "") as string,
            status: "completed" as const,
            fileUrl: (file.file_url || file.fileUrl) as string,
          })
        );
        setUploadedFiles(files);
      }
    } catch (error) {
      console.error("파일 목록 로드 실패:", error);
    }
  };

  // 파일 업로드 처리
  const handleFileUpload = async (file: File) => {
    const taskId = Date.now().toString();
    const isVideo = file.type.startsWith("video/");

    if (isVideo) {
      setUploadProgress({
        isUploading: true,
        message: "비디오 업로드 중...",
        progress: 10,
      });
    } else {
      setUploadProgress({
        isUploading: true,
        message: "이미지 업로드 중...",
        progress: 10,
      });
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("task_id", taskId);

    try {
      const response = await fetch("http://localhost:9090/api/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();

        setUploadProgress({
          isUploading: true,
          message: isVideo ? "비디오 프레임 분해 중..." : "이미지 처리 중...",
          progress: 50,
        });

        const newFile: UploadedFile = {
          id: taskId,
          fileName: file.name,
          originalName: file.name,
          fileType: file.type,
          fileSize: file.size,
          uploadedAt: new Date(),
          previewUrl: result.preview_url || "",
          status: "completed",
          fileUrl: result.file_url,
        };

        if (isVideo) {
          // 비디오 프레임 로드
          const framesResponse = await fetch(
            `http://localhost:9090/api/video-frames/${taskId}`
          );
          if (framesResponse.ok) {
            const framesData = await framesResponse.json();
            setVideoFrames(framesData.frame_urls);
            setVideoFps(framesData.fps || 30);
            setVideoDuration(framesData.total_frames / (framesData.fps || 30));
          }
        }

        setUploadProgress({
          isUploading: true,
          message: "완료!",
          progress: 100,
        });

        setTimeout(() => {
          setUploadProgress({
            isUploading: false,
            progress: 0,
            message: "",
          });
        }, 1000);

        setUploadedFiles((prev) => [...prev, newFile]);
      } else {
        console.error("업로드 실패:", response.statusText);
        setUploadProgress({
          isUploading: false,
          progress: 0,
          message: "",
        });
      }
    } catch (error) {
      console.error("업로드 오류:", error);
      setUploadProgress({
        isUploading: false,
        progress: 0,
        message: "",
      });
    }
  };

  // 파일 선택 처리
  const handleFileSelect = async (file: UploadedFile) => {
    setSelectedFile(file);
    setProcessHistory([]);

    if (file.fileType.startsWith("video/")) {
      try {
        const response = await fetch(
          `http://localhost:9090/api/video-frames/${file.id}`
        );
        if (response.ok) {
          const data = await response.json();
          setVideoFrames(data.frame_urls);
          setVideoFps(data.fps || 30);
          setVideoDuration(data.total_frames / (data.fps || 30));
          setSelectedFrameIndex(0);
        }
      } catch (error) {
        console.error("비디오 프레임 로드 실패:", error);
      }
    } else {
      // 이미지 모드
      if (file.previewUrl) {
        setCurrentImageUrl(file.previewUrl);
      }

      // 기존 처리 히스토리 로드
      try {
        const response = await fetch(`http://localhost:9090/api/files`);
        if (response.ok) {
          const data = await response.json();
          const fileData = data.files.find(
            (f: Record<string, unknown>) => f.id === file.id
          );
          if (fileData && fileData.process_history) {
            const history: ProcessStep[] = (
              fileData.process_history as Record<string, unknown>[]
            ).map((step: Record<string, unknown>) => ({
              id: step.id as string,
              action: step.action as string,
              description: step.description as string,
              resultUrl: step.resultUrl as string,
              timestamp: new Date(step.timestamp as string | number | Date),
              completedAt: new Date(step.completedAt as string | number | Date),
            }));
            setProcessHistory(history);
          }
        }
      } catch (error) {
        console.error("처리 히스토리 로드 실패:", error);
      }
    }
  };

  // 액션 실행
  const executeAction = async (actionId: string) => {
    if (!selectedFile || isProcessing) return;

    setIsProcessing(true);

    try {
      const backendTaskId = selectedFile.id;

      // 파일이 백엔드에 존재하는지 확인
      const checkResponse = await fetch(`http://localhost:9090/api/files`);
      if (!checkResponse.ok) {
        throw new Error("백엔드 파일 확인 실패");
      }

      const formData = new FormData();
      formData.append("task_id", backendTaskId);

      let endpoint = "";
      switch (actionId) {
        case "remove-background":
          endpoint = "/api/remove-background";
          break;
        case "upscale-x2":
          endpoint = "/api/upscale";
          formData.append("scale", "2");
          break;
        case "upscale-x4":
          endpoint = "/api/upscale";
          formData.append("scale", "4");
          break;
        case "vectorize-color":
          endpoint = "/api/vectorize";
          formData.append("color_mode", "color");
          break;
        case "vectorize-bw":
          endpoint = "/api/vectorize";
          formData.append("color_mode", "bw");
          break;
        case "style-retro":
          endpoint = "/api/style-transfer";
          formData.append("style", "retro");
          break;
        case "style-painting":
          endpoint = "/api/style-transfer";
          formData.append("style", "painting");
          break;
        case "extract-last-frame":
          endpoint = "/api/extract-last-frame";
          break;
        default:
          throw new Error("알 수 없는 액션입니다.");
      }

      const response = await fetch(`http://localhost:9090${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();

        // 스타일 전송의 경우 multiple results 처리
        if (actionId.startsWith("style-") && result.results) {
          const newSteps: ProcessStep[] = result.results.map(
            (res: Record<string, unknown>, index: number) => ({
              id: `${Date.now()}_${index}`,
              action: getActionLabel(actionId),
              description: `${getActionLabel(actionId)} 결과 ${index + 1}`,
              resultUrl: res.result_url as string,
              timestamp: new Date(),
              completedAt: new Date((res.completed_at as number) * 1000),
            })
          );

          setProcessHistory((prev) => [...prev, ...newSteps]);
          if (newSteps.length > 0) {
            setCurrentImageUrl(newSteps[0].resultUrl);
          }
        } else {
          // 단일 결과 처리
          const newStep: ProcessStep = {
            id: Date.now().toString(),
            action: getActionLabel(actionId),
            description: `${getActionLabel(actionId)} 처리 완료`,
            resultUrl: result.result_url as string,
            timestamp: new Date(),
            completedAt: new Date((result.completed_at as number) * 1000),
          };

          setProcessHistory((prev) => [...prev, newStep]);
          setCurrentImageUrl(result.result_url);
        }
      } else {
        console.error("처리 실패:", response.statusText);
      }
    } catch (error) {
      console.error("액션 실행 오류:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  // 액션 라벨 변환
  const getActionLabel = (actionId: string): string => {
    const labels: { [key: string]: string } = {
      "remove-background": "배경 제거",
      "upscale-x2": "2배 확대",
      "upscale-x4": "4배 확대",
      "vectorize-color": "컬러 벡터화",
      "vectorize-bw": "흑백 벡터화",
      "style-retro": "레트로 스타일",
      "style-painting": "페인팅 스타일",
      "extract-last-frame": "마지막 프레임 추출",
    };
    return labels[actionId] || actionId;
  };

  // 파일 삭제
  const deleteFile = async (fileId: string) => {
    try {
      const response = await fetch(
        `http://localhost:9090/api/delete-file/${fileId}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        setUploadedFiles((prev) => prev.filter((file) => file.id !== fileId));
        if (selectedFile?.id === fileId) {
          setSelectedFile(null);
          setCurrentImageUrl(null);
          setProcessHistory([]);
          setVideoFrames([]);
        }
      } else {
        console.error("파일 삭제 실패:", response.statusText);
      }
    } catch (error) {
      console.error("파일 삭제 오류:", error);
    }
  };

  // 프레임 선택/해제 토글
  const toggleFrameSelection = (frameIndex: number, e?: React.MouseEvent) => {
    if (e?.ctrlKey || e?.metaKey) {
      // Ctrl/Cmd 클릭: 다중 선택
      const newSelected = new Set(selectedFrames);
      if (newSelected.has(frameIndex)) {
        newSelected.delete(frameIndex);
      } else {
        newSelected.add(frameIndex);
      }
      setSelectedFrames(newSelected);
    } else {
      // 일반 클릭: 단일 선택
      const newSelected = new Set<number>();
      newSelected.add(frameIndex);
      setSelectedFrames(newSelected);
    }
  };

  // 선택된 프레임들 삭제
  const deleteSelectedFrames = async () => {
    if (selectedFrames.size === 0 || !selectedFile) return;

    const newFrames = videoFrames.filter(
      (_, index) => !selectedFrames.has(index)
    );

    // 백엔드에 프레임 삭제 반영
    try {
      const response = await fetch(
        `http://localhost:9090/api/video-frames/${selectedFile.id}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            frames: newFrames,
            operation: "delete",
            deletedFrames: Array.from(selectedFrames),
          }),
        }
      );

      if (response.ok) {
        setVideoFrames(newFrames);
        setSelectedFrames(new Set());

        // 현재 선택된 프레임이 삭제된 경우 첫 번째 프레임으로 이동
        if (selectedFrames.has(selectedFrameIndex) && newFrames.length > 0) {
          setSelectedFrameIndex(0);
          setCurrentImageUrl(newFrames[0]);
        }
      } else {
        console.error("프레임 삭제 실패:", response.statusText);
      }
    } catch (error) {
      console.error("프레임 삭제 중 오류:", error);
    }
  };

  // 타임라인 확대/축소
  const handleZoomChange = (delta: number) => {
    const newZoom = Math.max(0.3, Math.min(2.5, timelineZoom + delta));
    setTimelineZoom(newZoom);
  };

  // 프레임 선택
  const handleFrameSelect = (frameIndex: number) => {
    setSelectedFrameIndex(frameIndex);
    setCurrentImageUrl(videoFrames[frameIndex]);
  };

  // 타임라인 편집에 따른 비디오 재생 동기화
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isPlaying && videoFrames.length > 0) {
      interval = setInterval(() => {
        setSelectedFrameIndex((prevIndex) => {
          const nextIndex = (prevIndex + 1) % videoFrames.length;
          setCurrentImageUrl(videoFrames[nextIndex]);
          return nextIndex;
        });
      }, 100); // 10 FPS로 재생
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isPlaying, videoFrames]);

  // 모드 변경 시 워크스페이스 리셋
  useEffect(() => {
    setSelectedFile(null);
    setCurrentImageUrl(null);
    setProcessHistory([]);
    setVideoFrames([]);
    setSelectedFrameIndex(0);
    setIsProcessing(false);
  }, [currentMode]);

  // 다크모드 적용
  useEffect(() => {
    const newDarkMode = currentMode === "video";

    // HTML 클래스 업데이트 (Tailwind 다크모드)
    if (newDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [currentMode]);

  // 컴포넌트 마운트 시 데이터 로드
  useEffect(() => {
    loadFilesFromBackend();
  }, []);

  // 비디오 프레임 초기 프리로딩
  useEffect(() => {
    if (videoFrames.length > 0) {
      // 처음 10개 프레임을 프리로드
      for (let i = 0; i < Math.min(10, videoFrames.length); i++) {
        preloadFrame(i);
      }
      // 가시 범위 업데이트 - 새로운 타임라인에서는 불필요
    }
  }, [videoFrames.length, preloadFrame]); // eslint 경고 해결

  return (
    <SidebarProvider>
      <AppSidebar
        currentMode={currentMode}
        setCurrentMode={setCurrentMode}
        files={uploadedFiles}
        selectedFile={selectedFile}
        setSelectedFile={(file: UploadedFile | null) =>
          file && handleFileSelect(file)
        }
        onFileUpload={handleFileUpload}
        onDeleteFile={deleteFile}
      />
      <SidebarInset className="bg-background dark:bg-gray-900">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b px-4 bg-background dark:bg-gray-800 border-border dark:border-gray-700">
          <SidebarTrigger className="-ml-1 text-foreground dark:text-gray-300 hover:text-primary dark:hover:text-white" />
          <Separator
            orientation="vertical"
            className="mr-2 data-[orientation=vertical]:h-4 border-border dark:border-gray-600"
          />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbPage className="text-foreground dark:text-gray-300">
                  {currentMode === "image" ? "이미지 처리" : "비디오 처리"}
                </BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 w-full h-full bg-background dark:bg-gray-900">
          {selectedFile ? (
            selectedFile.fileType.startsWith("video/") ? (
              // 비디오 모드: 상하 레이아웃
              <div className="flex flex-col h-full w-full">
                {/* 상단: 비디오 플레이어 */}
                <div className="flex-1 flex items-center justify-center p-6 bg-background dark:bg-gray-900">
                  <div className="w-full h-full flex items-center justify-start pl-8">
                    {videoFrames.length > 0 ? (
                      // 프레임 기반 비디오 플레이어
                      <div className="relative w-full h-full flex items-center justify-center">
                        <img
                          src={
                            videoFrames[selectedFrameIndex] || videoFrames[0]
                          }
                          alt={`Frame ${selectedFrameIndex + 1}`}
                          className="max-w-full max-h-96 object-contain rounded-lg shadow-2xl border border-border dark:border-gray-700"
                          style={{ backgroundColor: "#000", maxWidth: "800px" }}
                        />
                        {/* 프레임 정보 오버레이 */}
                        <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-1 rounded text-sm">
                          Frame {selectedFrameIndex + 1} / {videoFrames.length}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-muted-foreground dark:text-gray-400">
                        <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
                        <p>비디오 프레임을 로딩 중입니다...</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* 하단: 다중 트랙 타임라인 */}
                {videoFrames.length > 0 && (
                  <Timeline
                    videoFrames={videoFrames}
                    selectedFrameIndex={selectedFrameIndex}
                    isPlaying={isPlaying}
                    selectedFrames={selectedFrames}
                    timelineZoom={timelineZoom}
                    videoFps={videoFps}
                    videoDuration={videoDuration}
                    onFrameSelect={(index) => {
                      setSelectedFrameIndex(index);
                      setCurrentImageUrl(videoFrames[index]);
                    }}
                    onPlayPause={togglePlayPause}
                    onStop={() => {
                      setIsPlaying(false);
                      setSelectedFrameIndex(0);
                      handleFrameSelect(0);
                    }}
                    onZoomChange={handleZoomChange}
                    onFrameToggleSelect={toggleFrameSelection}
                    onDeleteFrames={deleteSelectedFrames}
                    preloadedFrames={preloadedFrames}
                    onPreloadFrame={preloadFrame}
                  />
                )}

                {/* 기존 타임라인 */}
                {false && videoFrames.length > 0 && (
                  <div className="h-64 border-t bg-gray-900 flex-shrink-0">
                    <div className="h-full flex flex-col">
                      {/* 타임라인 헤더 */}
                      <div className="px-4 py-2 bg-gray-800 border-b border-gray-700 flex items-center justify-between flex-shrink-0">
                        <div className="flex items-center gap-4">
                          <div>
                            <h4 className="text-sm font-medium text-white">
                              타임라인
                            </h4>
                            <p className="text-xs text-gray-400">
                              {selectedFrameIndex + 1} / {videoFrames.length}{" "}
                              프레임
                              {selectedFrames.size > 0 &&
                                ` • ${selectedFrames.size}개 선택됨`}
                            </p>
                          </div>

                          {/* 재생 컨트롤 */}
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={togglePlayPause}
                              className="text-white hover:bg-gray-700 h-8 w-8 p-0"
                            >
                              {isPlaying ? (
                                <Pause className="h-4 w-4" />
                              ) : (
                                <Play className="h-4 w-4" />
                              )}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setIsPlaying(false);
                                setSelectedFrameIndex(0);
                                handleFrameSelect(0);
                              }}
                              className="text-white hover:bg-gray-700 h-8 w-8 p-0"
                            >
                              <Square className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {/* 확대/축소 */}
                          <div className="flex items-center gap-1 bg-gray-700 border border-gray-600 rounded-md p-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleZoomChange(-0.25)}
                              disabled={timelineZoom <= 0.5}
                              className="h-6 w-6 p-0 text-white hover:bg-gray-600"
                            >
                              <ZoomOut className="h-3 w-3" />
                            </Button>
                            <span className="text-xs px-2 min-w-[3rem] text-center font-mono text-gray-300">
                              {Math.round(timelineZoom * 100)}%
                            </span>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleZoomChange(0.25)}
                              disabled={timelineZoom >= 3}
                              className="h-6 w-6 p-0 text-white hover:bg-gray-600"
                            >
                              <ZoomIn className="h-3 w-3" />
                            </Button>
                          </div>

                          {/* 삭제 버튼 */}
                          {selectedFrames.size > 0 && (
                            <Button
                              variant="destructive"
                              size="sm"
                              onClick={deleteSelectedFrames}
                              className="h-7 bg-red-600 hover:bg-red-700"
                            >
                              <Trash2 className="h-3 w-3 mr-1" />
                              삭제 ({selectedFrames.size})
                            </Button>
                          )}
                        </div>
                      </div>

                      {/* 프레임 트랙 */}
                      <div className="flex-1 bg-gray-900 overflow-hidden">
                        <div className="h-full overflow-x-auto overflow-y-hidden">
                          <div className="flex h-full">
                            {/* 트랙 라벨 */}
                            <div className="w-16 bg-gray-800 border-r border-gray-700 flex items-center justify-center">
                              <Video className="h-4 w-4 text-gray-400" />
                            </div>

                            {/* 비디오 클립 */}
                            <div className="flex-1 p-2">
                              <div
                                className="h-full bg-gray-800 rounded-md border border-gray-700 relative overflow-x-auto overflow-y-hidden"
                                // onScroll 제거됨
                              >
                                {/* 전체 비디오 클립 배경 */}
                                <div
                                  className="h-full bg-cyan-600 rounded relative"
                                  style={{
                                    width: `${Math.max(
                                      300,
                                      videoFrames.length *
                                        Math.max(
                                          40,
                                          Math.round(80 * timelineZoom)
                                        )
                                    )}px`,
                                  }}
                                >
                                  {/* 프레임 타임라인 */}
                                  <div className="flex h-full">
                                    {videoFrames.map((frameUrl, index) => {
                                      const frameWidth = Math.max(
                                        40,
                                        Math.round(80 * timelineZoom)
                                      );
                                      const isSelected =
                                        selectedFrames.has(index);
                                      const isDragging = draggedFrame === index;

                                      return (
                                        <div
                                          key={index}
                                          className={`flex-shrink-0 cursor-pointer transition-all border-r border-gray-600 ${
                                            isDragging ? "opacity-50" : ""
                                          } ${
                                            selectedFrameIndex === index
                                              ? "ring-2 ring-blue-400"
                                              : ""
                                          }`}
                                          style={{
                                            width: frameWidth,
                                            minWidth: frameWidth,
                                          }}
                                          onClick={() =>
                                            optimizedPlayFrame(index)
                                          }
                                          onContextMenu={(e) => {
                                            e.preventDefault();
                                            toggleFrameSelection(index, e);
                                          }}
                                        >
                                          {/* 프레임 이미지 */}
                                          <div className="h-full relative">
                                            {preloadedFrames.has(index) ? (
                                              <img
                                                src={frameUrl}
                                                alt={`Frame ${index + 1}`}
                                                className="w-full h-full object-cover"
                                                draggable={false}
                                              />
                                            ) : (
                                              <div
                                                className="w-full h-full bg-gray-600 flex items-center justify-center"
                                                onMouseEnter={() =>
                                                  preloadFrame(index)
                                                }
                                              >
                                                <span className="text-xs text-gray-300">
                                                  {index + 1}
                                                </span>
                                              </div>
                                            )}

                                            {/* 선택된 프레임 표시 */}
                                            {isSelected && (
                                              <div className="absolute inset-0 bg-yellow-400/40"></div>
                                            )}

                                            {/* 프레임 번호 오버레이 */}
                                            <div className="absolute bottom-1 left-1 bg-black/70 text-white text-xs px-1 rounded">
                                              {index + 1}
                                            </div>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>

                                  {/* 플레이헤드 */}
                                  <div
                                    className="absolute top-0 bottom-0 w-0.5 bg-red-500 shadow-lg z-10 transition-all duration-100"
                                    style={{
                                      left: `${
                                        selectedFrameIndex *
                                          Math.max(
                                            40,
                                            Math.round(80 * timelineZoom)
                                          ) +
                                        Math.max(
                                          40,
                                          Math.round(80 * timelineZoom)
                                        ) /
                                          2
                                      }px`,
                                      transform: "translateX(-50%)",
                                    }}
                                  >
                                    {/* 플레이헤드 상단 핸들 */}
                                    <div className="absolute -top-1 left-1/2 transform -translate-x-1/2">
                                      <div className="w-3 h-3 bg-white border border-gray-400 rounded-full shadow-sm"></div>
                                    </div>
                                  </div>

                                  {/* 클립 정보 */}
                                  <div className="absolute top-1 left-2 text-white text-xs font-medium">
                                    Video Clip
                                  </div>
                                  <div className="absolute bottom-1 right-2 text-white text-xs opacity-75">
                                    {videoFrames.length} frames
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // 이미지 모드: 기존 좌우 레이아웃
              <div className="flex h-full w-full min-w-0">
                {/* 왼쪽 도구 패널 - 포토샵 스타일 */}
                <div className="w-16 border-r flex flex-col items-center py-2 gap-1 bg-muted/20 dark:bg-gray-800 border-border dark:border-gray-700">
                  {currentActions.map((action) => {
                    const Icon = action.icon;
                    return (
                      <Button
                        key={action.id}
                        variant="ghost"
                        size="sm"
                        onClick={() => executeAction(action.id)}
                        disabled={isProcessing}
                        className="w-12 h-12 p-0 flex flex-col items-center justify-center text-xs text-foreground dark:text-gray-300 hover:bg-muted dark:hover:bg-gray-700 dark:hover:text-white"
                        title={action.label}
                      >
                        {isProcessing ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <>
                            <Icon className="h-4 w-4" />
                            <span className="text-[10px] leading-tight mt-0.5 truncate w-full text-center">
                              {action.label}
                            </span>
                          </>
                        )}
                      </Button>
                    );
                  })}
                </div>

                {/* 메인 캔버스 영역 */}
                <div className="flex-1 flex flex-col relative min-w-0 overflow-hidden">
                  {/* 업로드 프로그레스 오버레이 */}
                  {uploadProgress.isUploading && (
                    <div className="absolute inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
                      <div className="bg-card p-6 rounded-lg shadow-lg border max-w-md w-full mx-4">
                        <div className="text-center mb-4">
                          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
                          <h3 className="font-semibold">
                            {uploadProgress.message}
                          </h3>
                        </div>
                        <Progress
                          value={uploadProgress.progress}
                          className="w-full"
                        />
                        <p className="text-sm text-muted-foreground text-center mt-2">
                          {uploadProgress.progress}%
                        </p>
                      </div>
                    </div>
                  )}

                  {/* 이미지인 경우: 기존 캔버스 */}
                  <div className="flex-1 flex items-center justify-center p-6 pb-2 min-h-0">
                    <div className="w-full max-w-full flex items-center justify-center h-[calc(100vh-8rem)]">
                      {currentImageUrl ? (
                        <img
                          src={currentImageUrl}
                          alt="Preview"
                          className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
                        />
                      ) : (
                        <div className="text-center text-muted-foreground">
                          <p>파일을 선택하면 여기에 표시됩니다</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* 오른쪽 처리 히스토리 패널 - 이미지인 경우에만 표시 */}
                <div className="w-80 border-l bg-muted/10 flex flex-col">
                  <div className="p-4 border-b">
                    <h3 className="text-lg font-semibold">처리 히스토리</h3>
                  </div>
                  <div className="flex-1 overflow-y-auto">
                    {processHistory.length > 0 ? (
                      <div className="p-2 space-y-2">
                        {processHistory.map((step) => (
                          <Card
                            key={step.id}
                            className="p-4 hover:bg-muted/50 transition-colors"
                          >
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex-1">
                                <h4 className="font-medium text-sm">
                                  {step.action}
                                </h4>
                                <p className="text-xs text-muted-foreground mt-1">
                                  {step.description}
                                </p>
                                <p className="text-xs text-muted-foreground mt-1">
                                  {step.completedAt.toLocaleString("ko-KR", {
                                    year: "numeric",
                                    month: "2-digit",
                                    day: "2-digit",
                                    hour: "2-digit",
                                    minute: "2-digit",
                                  })}
                                </p>
                              </div>
                              <div className="flex flex-col gap-1">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="h-7 px-2 text-xs"
                                  onClick={() => {
                                    window.open(
                                      `${step.resultUrl}?download=true`,
                                      "_blank"
                                    );
                                  }}
                                >
                                  <Download className="h-3 w-3 mr-1" />
                                  저장
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="h-7 px-2 text-xs"
                                  onClick={() => {
                                    setProcessHistory((prev) =>
                                      prev.filter((s) => s.id !== step.id)
                                    );
                                  }}
                                >
                                  <Trash2 className="h-3 w-3 mr-1" />
                                  삭제
                                </Button>
                              </div>
                            </div>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full text-center text-muted-foreground">
                        <div>
                          <div className="mb-2">📝</div>
                          <p className="text-sm">
                            처리 히스토리가
                            <br />
                            여기에 표시됩니다
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          ) : (
            <div className="flex flex-1 items-center justify-center">
              <div className="text-center text-muted-foreground">
                <div className="mb-4">
                  {currentMode === "image" ? (
                    <svg
                      className="mx-auto h-24 w-24"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1}
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="mx-auto h-24 w-24"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1}
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                  )}
                </div>
                <h3 className="text-lg font-semibold mb-2">
                  {currentMode === "image"
                    ? "이미지를 선택하세요"
                    : "비디오를 선택하세요"}
                </h3>
                <p className="text-sm">
                  왼쪽 사이드바에서 파일을 선택하거나 새로운 파일을
                  추가해보세요.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* 푸터 */}
        <footer className="border-t px-4 py-2 bg-background dark:bg-gray-800 border-border dark:border-gray-700">
          <div className="text-left text-xs text-muted-foreground dark:text-gray-400">
            © 2024 AICLUDE, Inc. All rights reserved.
          </div>
        </footer>
      </SidebarInset>
    </SidebarProvider>
  );
}
