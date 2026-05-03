package com.bot.projectlipsy;

import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ModelClient {
    private final Socket socket;
    private final OutputStream out;
    private final InputStream in;
    private int seq = 0;

    public ModelClient(String host, int port) throws IOException {
        this.socket = new Socket();
        this.socket.connect(new InetSocketAddress(host, port), 250);
        this.socket.setTcpNoDelay(true);
        this.socket.setSoTimeout(250);
        this.out = socket.getOutputStream();
        this.in = socket.getInputStream();
    }

    private ByteBuffer parseBytes(float[] batch) {
        int bodyLength = 4 + 8 + batch.length * 4;
        ByteBuffer byteBuffer = ByteBuffer.allocate(4 + bodyLength);

        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        byteBuffer.putInt(bodyLength);
        byteBuffer.putInt(seq);
        byteBuffer.putLong(System.nanoTime());
        for (float f : batch) {
            byteBuffer.putFloat(f);
        }
        return byteBuffer;
    }

    public void sendPacket(float[] batch) throws IOException {
        ByteBuffer batchBytes = parseBytes(batch);
        out.write(batchBytes.array());
        out.flush();
        seq++;
    }

    private byte[] readExact(int size) throws IOException {
        byte[] data = new byte[size];

        int offset = 0;
        while (offset < size) {
            int chunk = in.read(data, offset, size - offset);

            if (chunk == -1) {
                throw new EOFException("Socket closed.");
            }

            offset += chunk;
        }
        return data;
    }

    public float[] readPacket() throws IOException {
        // Header: int32 lenBytes
        byte[] lenBytes = readExact(4);
        int bodyLength = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();

        // Payload: int32 seq + int64 nano time + float32 action vector
        byte[] payloadBytes = readExact(bodyLength);
        ByteBuffer payloadBuffer = ByteBuffer.wrap(payloadBytes);
        payloadBuffer.order(ByteOrder.LITTLE_ENDIAN);
        int seq = payloadBuffer.getInt();
        long timeNs = payloadBuffer.getLong();

        int actionCount = (bodyLength - 12) / 4;

        float[] actions = new float[actionCount];

        for (int i = 0; i < actionCount; i++) {
            actions[i] = payloadBuffer.getFloat();
        }

        return actions;
    }

    public void stopConn() throws IOException {
        socket.close();
    }
}
