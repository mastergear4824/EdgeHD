"use client";

import * as React from "react";
import {
  Image as ImageIcon,
  Video,
  Plus,
  Trash2,
  User,
  Download,
} from "lucide-react";

import { Label } from "@/components/ui/label";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";

// Processing modes data
const processingModes = [
  {
    id: "image",
    title: "이미지 작업",
    icon: ImageIcon,
  },
  {
    id: "video",
    title: "비디오 작업",
    icon: Video,
  },
];

const userData = {
  name: "EdgeHD",
  email: "user@example.com",
  avatar: "/avatars/user.jpg",
};

// Types for our data
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

type ProcessingMode = "image" | "video";

interface AppSidebarProps extends React.ComponentProps<typeof Sidebar> {
  currentMode: ProcessingMode;
  setCurrentMode: (mode: ProcessingMode) => void;
  files: UploadedFile[];
  selectedFile: UploadedFile | null;
  setSelectedFile: (file: UploadedFile | null) => void;
  onFileUpload: (file: File) => Promise<void>;
  onDeleteFile: (fileId: string) => Promise<void>;
}

export function AppSidebar({
  currentMode,
  setCurrentMode,
  files,
  selectedFile,
  setSelectedFile,
  onFileUpload,
  onDeleteFile,
  ...props
}: AppSidebarProps) {
  const { setOpen } = useSidebar();
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // 현재 모드에 맞는 파일들만 필터링
  const filteredFiles = files.filter((file) => {
    if (currentMode === "image") {
      return file.fileType.startsWith("image/");
    } else {
      return file.fileType.startsWith("video/");
    }
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 60) return `${minutes}분 전`;
    if (hours < 24) return `${hours}시간 전`;
    if (days < 7) return `${days}일 전`;
    return date.toLocaleDateString();
  };

  return (
    <Sidebar
      collapsible="icon"
      className="overflow-hidden [&>[data-sidebar=sidebar]]:flex-row bg-background dark:bg-gray-900"
      {...props}
    >
      {/* 첫 번째 사이드바 - 작업 모드 선택 */}
      <Sidebar
        collapsible="none"
        className="!w-[calc(var(--sidebar-width-icon)_+_1px)] border-r bg-background dark:bg-gray-900 border-border dark:border-gray-700"
      >
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton size="lg" asChild className="md:h-8 md:p-0">
                <div className="cursor-pointer">
                  <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                    <img
                      src="/logo192.png"
                      alt="EdgeHD Logo"
                      className="size-8 object-contain"
                    />
                  </div>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-semibold text-foreground dark:text-white">
                      EdgeHD
                    </span>
                    <span className="truncate text-xs text-muted-foreground dark:text-gray-400">
                      AI Processing
                    </span>
                  </div>
                </div>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupContent className="px-1.5 md:px-0">
              <SidebarMenu>
                {processingModes.map((mode) => (
                  <SidebarMenuItem key={mode.id}>
                    <SidebarMenuButton
                      tooltip={{
                        children: mode.title,
                        hidden: false,
                      }}
                      onClick={() => {
                        setCurrentMode(mode.id as ProcessingMode);
                        setOpen(true);
                      }}
                      isActive={currentMode === mode.id}
                      className="px-2.5 md:px-2"
                    >
                      <mode.icon />
                      <span>{mode.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                size="lg"
                className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground md:h-8 md:p-0"
              >
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <User className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold text-foreground dark:text-white">
                    {userData.name}
                  </span>
                  <span className="truncate text-xs text-muted-foreground dark:text-gray-400">
                    {userData.email}
                  </span>
                </div>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
      </Sidebar>

      {/* 두 번째 사이드바 - 파일 리스트 */}
      <Sidebar
        collapsible="none"
        className="flex-1 bg-background dark:bg-gray-900"
      >
        <SidebarHeader className="gap-3.5 border-b p-4 bg-background dark:bg-gray-800 border-border dark:border-gray-700">
          <div className="flex w-full items-center justify-between">
            <div className="text-base font-medium text-foreground dark:text-white">
              {currentMode === "image" ? "이미지 작업" : "비디오 작업"}
            </div>
            <Label className="flex items-center gap-2 text-sm text-muted-foreground dark:text-gray-400">
              <span>{filteredFiles.length}개</span>
            </Label>
          </div>
          <div className="flex justify-center">
            <Button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 w-full bg-primary dark:bg-blue-600 hover:bg-primary/90 dark:hover:bg-blue-700 text-primary-foreground dark:text-white"
            >
              <Plus className="h-5 w-5" />새 작업 추가
            </Button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept={currentMode === "image" ? "image/*" : "video/*"}
            onChange={handleFileSelect}
          />
        </SidebarHeader>
        <SidebarContent className="bg-background dark:bg-gray-900">
          <SidebarGroup className="px-0">
            <SidebarGroupContent>
              {filteredFiles.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-8 text-center text-muted-foreground dark:text-gray-400">
                  <div className="mb-2">
                    {currentMode === "image" ? (
                      <ImageIcon className="h-12 w-12" />
                    ) : (
                      <Video className="h-12 w-12" />
                    )}
                  </div>
                  <p className="text-sm">
                    {currentMode === "image"
                      ? "이미지 파일을 추가해보세요"
                      : "비디오 파일을 추가해보세요"}
                  </p>
                </div>
              ) : (
                filteredFiles.map((file) => (
                  <div
                    key={file.id}
                    className={`flex items-center gap-3 border-b border-border dark:border-gray-700 p-3 hover:bg-muted dark:hover:bg-gray-800 cursor-pointer text-foreground dark:text-gray-200 ${
                      selectedFile?.id === file.id
                        ? "bg-muted dark:bg-gray-800"
                        : ""
                    }`}
                    onClick={() => setSelectedFile(file)}
                  >
                    {/* 미리보기 이미지 */}
                    <div className="flex-shrink-0">
                      {file.previewUrl ? (
                        <img
                          src={file.previewUrl}
                          alt="Preview"
                          className="w-12 h-12 object-cover rounded-lg border border-border dark:border-gray-600"
                        />
                      ) : (
                        <div className="w-12 h-12 bg-muted dark:bg-gray-700 rounded-lg flex items-center justify-center border border-border dark:border-gray-600">
                          <Video className="w-6 h-6 text-muted-foreground dark:text-gray-400" />
                        </div>
                      )}
                    </div>

                    {/* 파일 정보 */}
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-muted-foreground">
                        {formatFileSize(file.fileSize)} •{" "}
                        {formatDate(file.uploadedAt)}
                      </div>
                      {file.processingResults &&
                        file.processingResults.length > 0 && (
                          <div className="text-xs text-muted-foreground mt-1">
                            {file.processingResults.length}개 작업 완료
                          </div>
                        )}
                    </div>

                    {/* 액션 버튼들 */}
                    <div className="flex items-center gap-1">
                      {/* 최종 결과 다운로드 버튼 */}
                      {file.processingResults &&
                        file.processingResults.length > 0 && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={async (e) => {
                              e.stopPropagation();
                              const latestResult =
                                file.processingResults![
                                  file.processingResults!.length - 1
                                ];

                              try {
                                // 백엔드 API를 통해 파일 다운로드
                                const response = await fetch(
                                  `http://localhost:9090/api/image/${file.id}/result/${latestResult.id}`,
                                  { method: "GET" }
                                );

                                if (response.ok) {
                                  const blob = await response.blob();
                                  const url = window.URL.createObjectURL(blob);
                                  const link = document.createElement("a");
                                  link.href = url;
                                  link.download = `processed_${
                                    file.originalName || file.fileName
                                  }`;
                                  document.body.appendChild(link);
                                  link.click();
                                  document.body.removeChild(link);
                                  window.URL.revokeObjectURL(url);
                                } else {
                                  console.error(
                                    "다운로드 실패:",
                                    response.statusText
                                  );
                                }
                              } catch (error) {
                                console.error("다운로드 중 오류:", error);
                              }
                            }}
                            className="h-8 w-8 p-0 text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                          >
                            <Download className="h-4 w-4" />
                          </Button>
                        )}

                      {/* 삭제 버튼 */}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteFile(file.id);
                        }}
                        className="h-8 w-8 p-0 text-red-600 hover:text-red-700 hover:bg-red-50"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>
    </Sidebar>
  );
}
