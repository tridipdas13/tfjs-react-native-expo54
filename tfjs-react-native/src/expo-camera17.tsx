import React, { useCallback, useEffect, useRef, useState } from "react";
import { View } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import jpeg from "jpeg-js";

type Props = {
  onTensor: (t: tf.Tensor3D) => Promise<void> | void;
  intervalMs?: number;
  cameraProps?: React.ComponentProps<typeof CameraView>;
};

export function CameraTensorLoopExpo17({
  onTensor,
  intervalMs = 150,
  cameraProps,
}: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const camRef = useRef<any>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    (async () => {
      await tf.ready();
      if (!permission?.granted) await requestPermission();
      setRunning(true);
    })();
  }, [permission]);

  const tick = useCallback(async () => {
    if (!camRef.current) return;
    const pic = await camRef.current.takePictureAsync({
      base64: true,
      skipProcessing: true,
      quality: 0.1,
    }); // CameraView API for Expo 17 [web:1]
    if (!pic?.base64) return;

    const raw = Buffer.from(pic.base64, "base64");
    const { width, height, data } = jpeg.decode(raw, { useTArray: true });
    const rgba = new Uint8ClampedArray(data.buffer);
    const imageTensor = tf.tidy(() =>
      tf.browser.fromPixels({ data: rgba, width, height })
    ) as tf.Tensor3D; // tfjs RN supports fromPixels after tf.ready [web:12]

    try {
      await onTensor(imageTensor);
    } finally {
      imageTensor.dispose();
    }
  }, [onTensor]);

  useEffect(() => {
    if (!running) return;
    let id: any;
    const loop = async () => {
      await tick();
      id = setTimeout(loop, intervalMs);
    };
    loop();
    return () => clearTimeout(id);
  }, [running, tick, intervalMs]);

  if (!permission?.granted) return <View style={{ flex: 1 }} />;
  return <CameraView ref={camRef} style={{ flex: 1 }} {...cameraProps} />;
}
