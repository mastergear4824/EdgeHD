"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Play,
  Pause,
  Square,
  ZoomIn,
  ZoomOut,
  Trash2,
  Video,
} from "lucide-react";

interface TimelineProps {
  videoFrames: string[];
  selectedFrameIndex: number;
  isPlaying: boolean;
  selectedFrames: Set<number>;
  timelineZoom: number;
  videoFps: number;
  videoDuration: number;
  onFrameSelect: (index: number) => void;
  onPlayPause: () => void;
  onStop: () => void;
  onZoomChange: (delta: number) => void;
  onFrameToggleSelect: (index: number, event: React.MouseEvent) => void;
  onDeleteFrames: () => void;
  preloadedFrames: Set<number>;
  onPreloadFrame: (index: number) => void;
}

export default function Timeline({
  videoFrames,
  selectedFrameIndex,
  isPlaying,
  selectedFrames,
  timelineZoom,
  videoFps,
  videoDuration,
  onFrameSelect,
  onPlayPause,
  onStop,
  onZoomChange,
  onFrameToggleSelect,
  onDeleteFrames,
  preloadedFrames,
  onPreloadFrame,
}: TimelineProps) {
  // 안전한 초기값으로 설정
  const [containerWidth, setContainerWidth] = useState(800);
  const containerRef = useRef<HTMLDivElement>(null);

  // 실제 사용 가능한 너비를 동적으로 감지
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current && containerRef.current.parentElement) {
        // 부모 요소의 실제 너비를 측정
        const parentRect =
          containerRef.current.parentElement.getBoundingClientRect();
        const parentWidth = parentRect.width;
        const padding = 32; // 여백
        setContainerWidth(Math.max(600, parentWidth - padding));
      } else {
        // fallback: 화면 너비에서 예상 사이드바 너비를 뺀 값
        const sidebarWidth = 280;
        const padding = 32;
        const fallbackWidth = window.innerWidth - sidebarWidth - padding;
        setContainerWidth(Math.max(600, fallbackWidth));
      }
    };

    // 초기 실행
    updateWidth();

    // ResizeObserver로 부모 컨테이너 크기 변화 감지
    const resizeObserver = new ResizeObserver(() => {
      updateWidth();
    });

    // 부모 요소를 관찰
    if (containerRef.current && containerRef.current.parentElement) {
      resizeObserver.observe(containerRef.current.parentElement);
    }

    // 윈도우 리사이즈 이벤트도 함께 감지
    window.addEventListener("resize", updateWidth);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateWidth);
    };
  }, []);

  // 타임라인 트랙 라벨 영역 너비
  const trackLabelWidth = 80;

  // 실제 컨테이너 크기 기반으로 동적 계산
  const actualContainerWidth = containerRef.current
    ? containerRef.current.getBoundingClientRect().width
    : containerWidth;
  const availableWidth = actualContainerWidth - trackLabelWidth - 32;

  // 타임라인 콘텐츠 너비 계산
  const timelineContentWidth = Math.max(
    availableWidth, // 최소 너비
    availableWidth * timelineZoom // 줌에 따른 확장
  );
  const currentTime = selectedFrameIndex / videoFps;

  // 시간 눈금 생성
  const generateTimeMarkers = () => {
    const markers = [];
    // 줌 레벨에 따른 간격 조정
    let step: number;

    if (timelineZoom > 2) {
      step = 0.5; // 많이 확대했을 때는 0.5초 간격
    } else if (timelineZoom > 1) {
      step = 1; // 조금 확대했을 때는 1초 간격
    } else {
      step = 2; // 기본은 2초 간격
    }

    for (let time = 0; time <= videoDuration + step; time += step) {
      const position = (time / videoDuration) * timelineContentWidth;
      markers.push(
        <div
          key={time}
          className="absolute flex flex-col items-start"
          style={{ left: `${position}px` }}
        >
          <div className="w-0.5 h-3 bg-gray-400"></div>
          <span className="text-xs text-gray-400 mt-1 whitespace-nowrap">
            {time.toFixed(time % 1 === 0 ? 0 : 1)}s
          </span>
        </div>
      );
    }
    return markers;
  };

  return (
    <div
      ref={containerRef}
      className="h-80 border-t bg-gray-900 flex-shrink-0 overflow-hidden"
      style={{
        width: "100%",
        maxWidth: "100%",
        minWidth: "600px",
        flexShrink: 0,
        flexGrow: 1,
      }}
    >
      <div className="h-full flex flex-col">
        {/* 타임라인 헤더 */}
        <div className="px-4 py-2 bg-gray-800 border-b border-gray-700 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-4">
            <div>
              <h4 className="text-sm font-medium text-white">타임라인</h4>
              <p className="text-xs text-gray-400">
                {currentTime.toFixed(2)}s / {videoDuration.toFixed(2)}s •{" "}
                {videoFps}fps
              </p>
            </div>

            {/* 재생 컨트롤 */}
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={onPlayPause}
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
                onClick={onStop}
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
                onClick={() => onZoomChange(-0.25)}
                disabled={timelineZoom <= 0.3}
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
                onClick={() => onZoomChange(0.25)}
                disabled={timelineZoom >= 2.5}
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
                onClick={onDeleteFrames}
                className="h-7 bg-red-600 hover:bg-red-700"
              >
                <Trash2 className="h-3 w-3 mr-1" />
                삭제 ({selectedFrames.size})
              </Button>
            )}
          </div>
        </div>

        {/* 타임라인 본체 */}
        <div className="flex-1 flex flex-col">
          {/* 시간 눈금 영역 */}
          <div className="h-8 bg-gray-800 border-b border-gray-700 flex">
            {/* 트랙 라벨 영역 */}
            <div
              className="border-r border-gray-700 flex items-center justify-center flex-shrink-0"
              style={{ width: `${trackLabelWidth}px` }}
            >
              <span className="text-xs text-gray-400">Time</span>
            </div>

            {/* 시간 눈금 */}
            <div
              className="flex-1 relative overflow-x-auto"
              style={{ maxWidth: `${availableWidth}px` }}
            >
              <div
                className="h-full relative"
                style={{ width: `${timelineContentWidth}px` }}
              >
                {generateTimeMarkers()}

                {/* 현재 재생 위치 인디케이터 */}
                <div
                  className="absolute top-0 w-0.5 h-full bg-red-500 z-10"
                  style={{
                    left: `${
                      (currentTime / videoDuration) * timelineContentWidth
                    }px`,
                  }}
                >
                  <div className="absolute -top-1 left-1/2 transform -translate-x-1/2">
                    <div className="w-2 h-2 bg-red-500 rotate-45"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 트랙 영역 */}
          <div
            className="flex-1 flex"
            style={{
              maxWidth: "100%",
              width: "100%",
              overflow: "hidden",
            }}
          >
            {/* 트랙 라벨 영역 */}
            <div
              className="bg-gray-800 border-r border-gray-700 flex-shrink-0"
              style={{ width: `${trackLabelWidth}px` }}
            >
              {/* 비디오 트랙 */}
              <div className="h-20 border-b border-gray-700 flex items-center justify-center">
                <div className="flex flex-col items-center">
                  <Video className="h-4 w-4 text-cyan-400 mb-1" />
                  <span className="text-xs text-gray-400">Video</span>
                </div>
              </div>

              {/* 오디오 트랙 */}
              <div className="h-16 border-b border-gray-700 flex items-center justify-center">
                <div className="flex flex-col items-center">
                  <svg
                    className="h-3 w-3 text-green-400 mb-1"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M12 3v18l-4-4H4V7h4l4-4z" />
                    <path d="M17 7c0 1.3-.5 2.5-1.4 3.4" />
                    <path d="M19.5 4.5c2.9 2.9 2.9 7.6 0 10.5" />
                  </svg>
                  <span className="text-xs text-gray-400">Audio</span>
                </div>
              </div>
            </div>

            {/* 타임라인 컨텐츠 영역 */}
            <div
              className="flex-1 overflow-x-auto"
              style={{ maxWidth: `${availableWidth}px` }}
            >
              <div style={{ width: `${timelineContentWidth}px` }}>
                {/* 비디오 트랙 */}
                <div className="h-20 border-b border-gray-700 bg-gray-900 relative">
                  {/* 비디오 클립 */}
                  <div
                    className="absolute top-2 bottom-2 bg-cyan-600 rounded-md flex overflow-hidden"
                    style={{
                      left: "8px",
                      width: `${timelineContentWidth - 16}px`,
                    }}
                  >
                    {/* 프레임 썸네일들 */}
                    {videoFrames.map((frameUrl, index) => {
                      // 프레임 너비 계산
                      const frameContainerWidth = timelineContentWidth - 16;
                      const idealFrameWidth =
                        frameContainerWidth / videoFrames.length;
                      const minFrameWidth = 12;
                      const maxFrameWidth = 120;
                      const frameWidth = Math.max(
                        minFrameWidth,
                        Math.min(maxFrameWidth, idealFrameWidth)
                      );
                      const isSelected = selectedFrames.has(index);
                      const isCurrentFrame = selectedFrameIndex === index;

                      return (
                        <div
                          key={index}
                          className={`flex-shrink-0 h-full cursor-pointer border-r border-cyan-500/30 relative hover:border-cyan-300 ${
                            isCurrentFrame
                              ? "ring-1 ring-yellow-400 ring-inset"
                              : ""
                          }`}
                          style={{
                            width: frameWidth,
                            minWidth: `${minFrameWidth}px`,
                          }}
                          onClick={() => onFrameSelect(index)}
                          onContextMenu={(e) => {
                            e.preventDefault();
                            onFrameToggleSelect(index, e);
                          }}
                        >
                          {frameWidth > 24 && preloadedFrames.has(index) ? (
                            <img
                              src={frameUrl}
                              alt={`Frame ${index + 1}`}
                              className="w-full h-full object-cover"
                              draggable={false}
                            />
                          ) : (
                            <div
                              className="w-full h-full bg-cyan-700 flex items-center justify-center"
                              onMouseEnter={() => onPreloadFrame(index)}
                            >
                              {frameWidth > 20 && (
                                <span className="text-xs text-white/60">
                                  {index + 1}
                                </span>
                              )}
                            </div>
                          )}

                          {/* 선택된 프레임 표시 */}
                          {isSelected && (
                            <div className="absolute inset-0 bg-yellow-400/50"></div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* 오디오 트랙 */}
                <div className="h-16 border-b border-gray-700 bg-gray-900 relative">
                  {/* 오디오 파형 */}
                  <div
                    className="absolute top-2 bottom-2 bg-green-600 rounded-sm flex items-center justify-center"
                    style={{
                      left: "8px",
                      width: `${timelineContentWidth - 16}px`,
                    }}
                  >
                    <span className="text-xs text-white/80">
                      Audio Track ({videoDuration.toFixed(1)}s)
                    </span>
                  </div>
                </div>

                {/* 텍스트/이펙트 트랙 */}
                <div className="h-12 border-b border-gray-700 bg-gray-900 relative">
                  <div className="absolute top-2 bottom-2 left-2 right-2 border-2 border-dashed border-gray-600 rounded flex items-center justify-center">
                    <span className="text-xs text-gray-500">
                      텍스트 & 이펙트
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
