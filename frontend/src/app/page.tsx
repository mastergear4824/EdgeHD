"use client";

import { useState } from "react";
import {
  Upload,
  Wand2,
  Zap,
  Palette,
  Video,
  Image as ImageIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";

interface ProcessingState {
  isProcessing: boolean;
  progress: number;
  message: string;
  result?: string;
}

export default function Home() {
  const [processingState, setProcessingState] = useState<ProcessingState>({
    isProcessing: false,
    progress: 0,
    message: "",
  });

  const handleFileUpload = async (file: File, operation: string) => {
    setProcessingState({
      isProcessing: true,
      progress: 0,
      message: "파일 업로드 중...",
    });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("work_id", crypto.randomUUID());

    try {
      const response = await fetch(`http://localhost:8080/api/${operation}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("처리 실패");
      }

      const result = await response.json();

      setProcessingState({
        isProcessing: false,
        progress: 100,
        message: "처리 완료!",
        result: result.download_url,
      });

      toast.success("처리가 완료되었습니다!");
    } catch (error) {
      setProcessingState({
        isProcessing: false,
        progress: 0,
        message: "처리 실패",
      });
      toast.error("처리 중 오류가 발생했습니다.");
    }
  };

  const handleDrop = (e: React.DragEvent, operation: string) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0], operation);
    }
  };

  const handleFileSelect = (
    e: React.ChangeEvent<HTMLInputElement>,
    operation: string
  ) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0], operation);
    }
  };

  const UploadCard = ({
    title,
    description,
    icon: Icon,
    operation,
    accept = "image/*",
  }: {
    title: string;
    description: string;
    icon: any;
    operation: string;
    accept?: string;
  }) => (
    <Card className="group hover:shadow-lg transition-all duration-300">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className="h-5 w-5 text-primary" />
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer"
          onDrop={(e) => handleDrop(e, operation)}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById(`file-${operation}`)?.click()}
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-sm text-gray-600 mb-2">
            파일을 드래그하거나 클릭하여 업로드
          </p>
          <p className="text-xs text-gray-400">
            {accept === "image/*"
              ? "PNG, JPG, JPEG, GIF, BMP, WebP"
              : "MP4, AVI, MOV, MKV"}
          </p>
          <input
            id={`file-${operation}`}
            type="file"
            accept={accept}
            className="hidden"
            onChange={(e) => handleFileSelect(e, operation)}
          />
        </div>

        {processingState.isProcessing && (
          <div className="mt-4">
            <Progress value={processingState.progress} className="mb-2" />
            <p className="text-sm text-gray-600">{processingState.message}</p>
          </div>
        )}

        {processingState.result && (
          <div className="mt-4">
            <Button asChild className="w-full">
              <a
                href={`http://localhost:8080${processingState.result}`}
                download
              >
                결과 다운로드
              </a>
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">EdgeHD</h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            AI 기반 이미지 및 비디오 처리 플랫폼
          </p>
          <p className="text-sm text-gray-500 mt-2">
            배경 제거, 업스케일링, 벡터화를 한 번에
          </p>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="image" className="max-w-6xl mx-auto">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="image" className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4" />
              이미지 처리
            </TabsTrigger>
            <TabsTrigger value="video" className="flex items-center gap-2">
              <Video className="h-4 w-4" />
              비디오 처리
            </TabsTrigger>
          </TabsList>

          <TabsContent value="image">
            <div className="grid md:grid-cols-3 gap-6">
              <UploadCard
                title="배경 제거"
                description="AI가 자동으로 배경을 제거합니다"
                icon={Wand2}
                operation="upload"
              />
              <UploadCard
                title="이미지 업스케일링"
                description="AI로 이미지 해상도를 2x, 4x 향상"
                icon={Zap}
                operation="upscale"
              />
              <UploadCard
                title="벡터화"
                description="이미지를 SVG 벡터 형식으로 변환"
                icon={Palette}
                operation="vectorize"
              />
            </div>
          </TabsContent>

          <TabsContent value="video">
            <div className="grid md:grid-cols-2 gap-6">
              <UploadCard
                title="비디오 배경 제거"
                description="비디오의 각 프레임에서 배경을 제거"
                icon={Video}
                operation="process_video"
                accept="video/*"
              />
              <UploadCard
                title="마지막 프레임 추출"
                description="비디오에서 마지막 프레임을 이미지로 추출"
                icon={ImageIcon}
                operation="extract_last_frame"
                accept="video/*"
              />
            </div>
          </TabsContent>
        </Tabs>

        {/* Features */}
        <div className="mt-16 grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="bg-blue-100 rounded-full p-3 w-12 h-12 mx-auto mb-4">
              <Wand2 className="h-6 w-6 text-blue-600" />
            </div>
            <h3 className="font-semibold mb-2">AI 기반 처리</h3>
            <p className="text-sm text-gray-600">최신 AI 모델로 고품질 결과</p>
          </div>
          <div className="text-center">
            <div className="bg-green-100 rounded-full p-3 w-12 h-12 mx-auto mb-4">
              <Zap className="h-6 w-6 text-green-600" />
            </div>
            <h3 className="font-semibold mb-2">빠른 처리</h3>
            <p className="text-sm text-gray-600">GPU 가속으로 빠른 처리 속도</p>
          </div>
          <div className="text-center">
            <div className="bg-purple-100 rounded-full p-3 w-12 h-12 mx-auto mb-4">
              <Palette className="h-6 w-6 text-purple-600" />
            </div>
            <h3 className="font-semibold mb-2">다양한 형식</h3>
            <p className="text-sm text-gray-600">이미지, 비디오, 벡터 지원</p>
          </div>
          <div className="text-center">
            <div className="bg-orange-100 rounded-full p-3 w-12 h-12 mx-auto mb-4">
              <Upload className="h-6 w-6 text-orange-600" />
            </div>
            <h3 className="font-semibold mb-2">간편한 사용</h3>
            <p className="text-sm text-gray-600">드래그 앤 드롭으로 간편하게</p>
          </div>
        </div>
      </div>
    </div>
  );
}
