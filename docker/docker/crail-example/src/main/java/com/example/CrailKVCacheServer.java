package com.example;

import org.apache.crail.*;
import org.apache.crail.conf.CrailConfiguration;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

public class CrailKVCacheServer {
    private static final int BUFFER_SIZE = 8 * 1024 * 1024; // 8MB buffer
    private static final int DEFAULT_PORT = 9876;
    private static final int DEFAULT_THREADS = 10;
    private static final int CLIENT_TIMEOUT_MS = 300000; // 5分钟
    
    // Crail连接实例
    private static CrailStore store = null;
    private static final Object storeLock = new Object();
    
    // 线程池处理客户端请求
    private static ExecutorService threadPool;
    
    // 用于优雅关闭的标志
    private static AtomicBoolean running = new AtomicBoolean(true);
    
    // 统计信息
    private static int connectionCount = 0;
    private static AtomicLong totalBytesUploaded = new AtomicLong(0);
    private static AtomicLong totalBytesDownloaded = new AtomicLong(0);
    
    public static void main(String[] args) {
        int port = DEFAULT_PORT;
        int threads = DEFAULT_THREADS;
        String confDir = "/root/crail/conf";
        
        // 解析命令行参数
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--port") && i + 1 < args.length) {
                port = Integer.parseInt(args[i + 1]);
                i++;
            } else if (args[i].equals("--threads") && i + 1 < args.length) {
                threads = Integer.parseInt(args[i + 1]);
                i++;
            } else if (args[i].equals("--conf") && i + 1 < args.length) {
                confDir = args[i + 1];
                i++;
            }
        }
        
        // 设置Crail配置目录
        System.setProperty("crail.conf.dir", confDir);
        
        // 初始化Crail连接
        try {
            initCrailStore();
            System.err.println("Crail connection initialized successfully");
        } catch (Exception e) {
            System.err.println("Failed to initialize Crail connection: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
        
        // 创建线程池
        threadPool = Executors.newFixedThreadPool(threads);
        System.err.println("Created thread pool with " + threads + " worker threads");
        
        // 添加关闭钩子
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.err.println("Shutting down server...");
            shutdownServer();
        }));
        
        // 启动服务器
        ServerSocket serverSocket = null;
        try {
            serverSocket = new ServerSocket(port);
            serverSocket.setReuseAddress(true);
            System.err.println("CrailKVCacheServer started on port " + port);
            System.err.println("Ready to accept connections");
            
            while (running.get()) {
                try {
                    // 接受新的连接
                    Socket clientSocket = serverSocket.accept();
                    int clientId = ++connectionCount;
                    
                    // 提交到线程池处理
                    threadPool.submit(() -> handleClient(clientSocket, clientId));
                } catch (IOException e) {
                    if (running.get()) {
                        System.err.println("Error accepting connection: " + e.getMessage());
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Server socket error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 关闭服务器套接字
            if (serverSocket != null && !serverSocket.isClosed()) {
                try {
                    serverSocket.close();
                } catch (IOException e) {
                    System.err.println("Error closing server socket: " + e.getMessage());
                }
            }
            
            // 确保资源被释放
            shutdownServer();
        }
    }
    
    private static void shutdownServer() {
        // 设置关闭标志
        running.set(false);
        
        // 关闭线程池
        if (threadPool != null && !threadPool.isShutdown()) {
            System.err.println("Shutting down thread pool...");
            threadPool.shutdown();
            try {
                // 等待所有任务完成
                if (!threadPool.awaitTermination(30, TimeUnit.SECONDS)) {
                    System.err.println("Thread pool did not terminate gracefully, forcing shutdown");
                    threadPool.shutdownNow();
                }
            } catch (InterruptedException e) {
                threadPool.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        // 关闭Crail连接
        try {
            closeCrailStore();
        } catch (Exception e) {
            System.err.println("Error closing Crail store: " + e.getMessage());
        }
        
        // 打印统计信息
        System.err.println("\nServer Statistics:");
        System.err.println("Total connections handled: " + connectionCount);
        System.err.println("Total data uploaded: " + formatSize(totalBytesUploaded.get()));
        System.err.println("Total data downloaded: " + formatSize(totalBytesDownloaded.get()));
        
        System.err.println("Server shutdown complete");
    }
    
    private static void handleClient(Socket clientSocket, int connectionId) {
        String clientAddress = clientSocket.getRemoteSocketAddress().toString();
        System.err.println("Client #" + connectionId + " connected from " + clientAddress);
        
        DataInputStream in = null;
        DataOutputStream out = null;
        
        try {
            // 设置超时
            clientSocket.setSoTimeout(CLIENT_TIMEOUT_MS);
            clientSocket.setTcpNoDelay(true);
            clientSocket.setKeepAlive(true);
            
            // 创建输入输出流
            in = new DataInputStream(new BufferedInputStream(clientSocket.getInputStream(), BUFFER_SIZE));
            out = new DataOutputStream(new BufferedOutputStream(clientSocket.getOutputStream(), BUFFER_SIZE));
            
            // 读取操作类型
            int cmdLength = in.readUnsignedShort(); // 读取2字节长度
            if (cmdLength > 1024) {
                throw new IOException("Command length too large: " + cmdLength);
            }
            
            byte[] cmdBytes = new byte[cmdLength];
            in.readFully(cmdBytes); // 确保读取完整命令
            String operation = new String(cmdBytes, "UTF-8");
            
            System.err.println("Client #" + connectionId + " requested operation: " + operation);
            
            switch (operation.toLowerCase()) {
                case "upload":
                    handleUpload(in, out, connectionId);
                    break;
                    
                case "download":
                    handleDownload(in, out, connectionId);
                    break;
                    
                case "ping":
                    handlePing(out, connectionId);
                    break;
                    
                case "shutdown":
                    handleShutdown(out, connectionId);
                    break;
                    
                default:
                    System.err.println("Client #" + connectionId + " requested unknown operation: " + operation);
                    sendError(out, "Unknown operation: " + operation);
            }
        } catch (EOFException e) {
            System.err.println("Client #" + connectionId + " closed connection unexpectedly");
        } catch (SocketTimeoutException e) {
            System.err.println("Client #" + connectionId + " connection timed out: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Client #" + connectionId + " I/O error: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Error handling client #" + connectionId + ": " + e.getMessage());
            e.printStackTrace();
            
            // 尝试发送错误响应
            if (out != null) {
                try {
                    sendError(out, e.getMessage() != null ? e.getMessage() : "Internal server error");
                } catch (Exception ex) {
                    // 忽略发送错误响应时的异常
                }
            }
        } finally {
            // 关闭流
            if (in != null) {
                try {
                    in.close();
                } catch (IOException e) {
                    // 忽略关闭流时的异常
                }
            }
            
            if (out != null) {
                try {
                    out.close();
                } catch (IOException e) {
                    // 忽略关闭流时的异常
                }
            }
            
            // 关闭客户端套接字
            try {
                if (!clientSocket.isClosed()) {
                    clientSocket.close();
                }
                System.err.println("Client #" + connectionId + " disconnected");
            } catch (IOException e) {
                System.err.println("Error closing client #" + connectionId + " socket: " + e.getMessage());
            }
        }
    }
    
    private static void handleUpload(DataInputStream in, DataOutputStream out, int connectionId) throws Exception {
        long startTime = System.currentTimeMillis();
        CrailBufferedOutputStream outstream = null;
        CrailNode node = null;
        
        try {
            // 读取Crail路径长度和路径
            int pathLength = in.readUnsignedShort();
            if (pathLength > 4096) {
                throw new IOException("Path length too large: " + pathLength);
            }
            
            byte[] pathBytes = new byte[pathLength];
            in.readFully(pathBytes);
            String crailPath = new String(pathBytes, "UTF-8");
            
            // 读取数据大小
            long dataSize = in.readLong();
            if (dataSize < 0 || dataSize > 10L * 1024 * 1024 * 1024) { // 限制为10GB
                throw new IOException("Invalid data size: " + dataSize);
            }
            
            System.err.println("Client #" + connectionId + " uploading to: " + crailPath + 
                               " (size: " + formatSize(dataSize) + ")");
            
            // 获取Crail连接
            CrailStore crailStore = getCrailStore();
            
            // 创建父目录
            createParentDirectories(crailStore, crailPath);
            
            // 检查文件是否已存在，如果存在则先删除
            try {
                node = crailStore.lookup(crailPath).get();
                if (node != null) {
                    System.err.println("File already exists, deleting: " + crailPath);
                    crailStore.delete(crailPath, false).get();
                }
            } catch (Exception e) {
                // 文件不存在或删除过程中出错，继续尝试创建
                System.err.println("File does not exist or could not be deleted: " + e.getMessage());
            }
            
            // 创建Crail文件
            try {
                node = crailStore.create(crailPath, CrailNodeType.DATAFILE,
                        CrailStorageClass.DEFAULT, CrailLocationClass.DEFAULT, false).get();
                node.syncDir();
            } catch (Exception e) {
                System.err.println("Failed to create file " + crailPath + ": " + e.getMessage());
                throw new IOException("Cannot create Crail file: " + e.getMessage(), e);
            }
            
            // 获取Crail文件
            CrailFile crailFile = node.asFile();
            
            // 创建输出流并设置较大预分配
            try {
                System.err.println("Creating output stream for file: " + crailPath);
                outstream = crailFile.getBufferedOutputStream(dataSize);
                
                // 使用缓冲区传输数据
                byte[] buffer = new byte[BUFFER_SIZE];
                long bytesRemaining = dataSize;
                int bytesRead;
                long lastProgressTime = System.currentTimeMillis();
                long totalBytesRead = 0;
                
                System.err.println("Starting to read data from client #" + connectionId);
                
                while (bytesRemaining > 0 && (bytesRead = in.read(buffer, 0, (int) Math.min(buffer.length, bytesRemaining))) != -1) {
                    if (bytesRead > 0) {
                        // 写入数据到Crail
                        outstream.write(buffer, 0, bytesRead);
                        
                        bytesRemaining -= bytesRead;
                        totalBytesRead += bytesRead;
                        totalBytesUploaded.addAndGet(bytesRead);
                        
                        // 每隔一定大小强制刷新
                        if (totalBytesRead % (32 * 1024 * 1024) == 0) {
                            outstream.flush();
                        }
                        
                        // 每秒最多打印一次进度
                        long now = System.currentTimeMillis();
                        if (now - lastProgressTime >= 1000) {
                            double progress = 100.0 * totalBytesRead / dataSize;
                            System.err.printf("Client #%d upload progress: %.1f%% (%s/%s)\n", 
                                             connectionId, progress, 
                                             formatSize(totalBytesRead), 
                                             formatSize(dataSize));
                            lastProgressTime = now;
                        }
                    }
                }
                
                if (bytesRemaining > 0) {
                    throw new IOException("Incomplete upload: missing " + bytesRemaining + " bytes");
                }
                
                // 确保所有数据都同步到存储
                System.err.println("Flushing data for client #" + connectionId);
                outstream.flush();
                
                // 关闭输出流之前，确保所有操作完成
                System.err.println("Syncing data for client #" + connectionId);
                outstream.sync();
                
                // 显式等待一段时间，确保异步操作完成
                Thread.sleep(200);
                
                // 现在安全地关闭流
                System.err.println("Closing output stream for client #" + connectionId);
                safeCloseStream(outstream, connectionId);
                outstream = null;
                
                // 再次同步目录以确保所有元数据更新
                node.syncDir();
                
                long endTime = System.currentTimeMillis();
                double duration = (endTime - startTime) / 1000.0;
                double speedMBps = (dataSize / (1024.0 * 1024.0)) / duration;
                
                System.err.println("Client #" + connectionId + " upload completed");
                System.err.printf("Size: %s, Time: %.2f sec, Speed: %.2f MB/s\n",
                                 formatSize(dataSize), duration, speedMBps);
                
                // 发送成功响应
                sendOK(out);
                
            } catch (Exception e) {
                System.err.println("Error during client #" + connectionId + " upload: " + e.getMessage());
                e.printStackTrace();
                
                // 尝试关闭流
                if (outstream != null) {
                    safeCloseStream(outstream, connectionId);
                    outstream = null;
                }
                
                // 尝试删除不完整的文件
                try {
                    System.err.println("Attempting to delete incomplete file: " + crailPath);
                    crailStore.delete(crailPath, false).get();
                } catch (Exception ex) {
                    System.err.println("Warning: Failed to delete incomplete file: " + ex.getMessage());
                }
                
                // 发送错误响应
                sendError(out, e.getMessage());
                throw e;
            }
        } catch (Exception e) {
            System.err.println("Outer error handling for client #" + connectionId + ": " + e.getMessage());
            e.printStackTrace();
            
            // 确保错误被传递到客户端
            try {
                sendError(out, "Server error: " + e.getMessage());
            } catch (Exception ex) {
                // 忽略发送错误时的异常
            }
            
            throw e;
        } finally {
            // 确保资源被释放
            if (outstream != null) {
                try {
                    safeCloseStream(outstream, connectionId);
                } catch (Exception e) {
                    System.err.println("Warning: Error in finally block closing output stream: " + e.getMessage());
                }
            }
        }
    }
    
    /**
     * 安全关闭 CrailBufferedOutputStream，处理和记录任何错误
     */
    private static void safeCloseStream(CrailBufferedOutputStream stream, int connectionId) {
        if (stream != null) {
            try {
                // 首先尝试刷新
                System.err.println("Flushing stream for client #" + connectionId);
                stream.flush();
                
                // 然后尝试同步
                System.err.println("Syncing stream for client #" + connectionId);
                stream.sync();
                
                // 等待一段时间，确保异步操作完成
                Thread.sleep(200);
                
                // 最后关闭
                System.err.println("Closing stream for client #" + connectionId);
                stream.close();
            } catch (Exception e) {
                System.err.println("Warning: Error during safe stream close: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private static void handleDownload(DataInputStream in, DataOutputStream out, int connectionId) throws Exception {
        long startTime = System.currentTimeMillis();
        CrailBufferedInputStream instream = null;
        
        try {
            // 读取Crail路径长度和路径
            int pathLength = in.readUnsignedShort();
            if (pathLength > 4096) {
                throw new IOException("Path length too large: " + pathLength);
            }
            
            byte[] pathBytes = new byte[pathLength];
            in.readFully(pathBytes);
            String crailPath = new String(pathBytes, "UTF-8");
            
            System.err.println("Client #" + connectionId + " downloading from: " + crailPath);
            
            // 获取Crail连接
            CrailStore crailStore = getCrailStore();
            
            // 查找文件
            CrailNode node = crailStore.lookup(crailPath).get();
            if (node.getType() != CrailNodeType.DATAFILE) {
                throw new FileNotFoundException("Not a regular file: " + crailPath);
            }
            
            // 获取文件
            CrailFile file = node.asFile();
            long fileSize = file.getCapacity();
            
            System.err.println("Client #" + connectionId + " downloading file: " + crailPath + 
                               " (size: " + formatSize(fileSize) + ")");
            
            // 发送成功响应和文件大小
            out.writeUTF("OK");
            out.writeLong(fileSize);
            out.flush();
            
            // 从Crail读取并发送数据
            try {
                instream = file.getBufferedInputStream(fileSize);
                
                byte[] buffer = new byte[BUFFER_SIZE];
                long bytesRemaining = fileSize;
                int bytesRead;
                long lastProgressTime = System.currentTimeMillis();
                
                while (bytesRemaining > 0 && (bytesRead = instream.read(buffer, 0, (int) Math.min(buffer.length, bytesRemaining))) > 0) {
                    out.write(buffer, 0, bytesRead);
                    bytesRemaining -= bytesRead;
                    totalBytesDownloaded.addAndGet(bytesRead);
                    
                    // 定期刷新缓冲区确保数据发送
                    if (bytesRemaining % (50 * 1024 * 1024) < BUFFER_SIZE) {
                        out.flush();
                    }
                    
                    // 每秒最多打印一次进度
                    long now = System.currentTimeMillis();
                    if (now - lastProgressTime >= 1000) {
                        double progress = 100.0 * (fileSize - bytesRemaining) / fileSize;
                        System.err.printf("Client #%d download progress: %.1f%% (%s/%s)\n", 
                                         connectionId, progress, 
                                         formatSize(fileSize - bytesRemaining), 
                                         formatSize(fileSize));
                        lastProgressTime = now;
                    }
                }
                
                // 确保所有数据都发送出去
                out.flush();
                
                if (bytesRemaining > 0) {
                    throw new IOException("Incomplete download: missing " + bytesRemaining + " bytes");
                }
                
                // 安全关闭输入流
                if (instream != null) {
                    instream.close();
                    instream = null;
                }
                
                long endTime = System.currentTimeMillis();
                double duration = (endTime - startTime) / 1000.0;
                double speedMBps = (fileSize / (1024.0 * 1024.0)) / duration;
                
                System.err.println("Client #" + connectionId + " download completed");
                System.err.printf("Size: %s, Time: %.2f sec, Speed: %.2f MB/s\n",
                                 formatSize(fileSize), duration, speedMBps);
            } finally {
                // 确保资源被释放
                if (instream != null) {
                    try {
                        instream.close();
                    } catch (Exception e) {
                        System.err.println("Warning: Error closing input stream: " + e.getMessage());
                    }
                }
            }
            
        } catch (Exception e) {
            System.err.println("Error during client #" + connectionId + " download: " + e.getMessage());
            
            // 发送错误响应（如果还没有发送正确的响应）
            if (startTime == System.currentTimeMillis()) { // 近似检查是否已经发送了成功响应
                sendError(out, e.getMessage());
            }
            throw e;
        }
    }
    
    private static void handlePing(DataOutputStream out, int connectionId) throws IOException {
        System.err.println("Client #" + connectionId + " ping request");
        out.writeUTF("PONG");
        out.flush();
        System.err.println("Client #" + connectionId + " ping response sent");
    }
    
    private static void handleShutdown(DataOutputStream out, int connectionId) throws IOException {
        System.err.println("Client #" + connectionId + " requested server shutdown");
        out.writeUTF("OK");
        out.flush();
        
        // 延迟关闭，确保响应已发送
        new Thread(() -> {
            try {
                Thread.sleep(500);
                running.set(false);
                System.err.println("Server shutdown initiated by client #" + connectionId);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }
    
    private static void sendOK(DataOutputStream out) throws IOException {
        out.writeUTF("OK");
        out.flush();
    }
    
    private static void sendError(DataOutputStream out, String message) throws IOException {
        out.writeUTF("ERROR");
        out.writeUTF(message != null ? message : "Unknown error");
        out.flush();
    }
    
    private static void createParentDirectories(CrailStore store, String path) throws Exception {
        String[] parts = path.split("/");
        StringBuilder currentPath = new StringBuilder();
        
        for (int i = 0; i < parts.length - 1; i++) {
            if (parts[i].isEmpty()) continue;
            
            currentPath.append("/").append(parts[i]);
            String dirPath = currentPath.toString();
            
            try {
                CrailNode node = store.lookup(dirPath).get();
                if (node.getType() != CrailNodeType.DIRECTORY) {
                    throw new IOException("Path exists but is not a directory: " + dirPath);
                }
            } catch (Exception e) {
                try {
                    store.create(dirPath, CrailNodeType.DIRECTORY,
                            CrailStorageClass.DEFAULT, CrailLocationClass.DEFAULT, false).get();
                    System.err.println("Created directory: " + dirPath);
                } catch (Exception ex) {
                    // 检查目录是否已经存在（可能由另一个进程创建）
                    try {
                        CrailNode node = store.lookup(dirPath).get();
                        if (node.getType() != CrailNodeType.DIRECTORY) {
                            throw new IOException("Failed to create directory: " + dirPath);
                        }
                    } catch (Exception innerEx) {
                        throw new IOException("Could not create or verify directory: " + dirPath, innerEx);
                    }
                }
            }
        }
    }
    
    // 初始化Crail连接
    private static void initCrailStore() throws Exception {
        synchronized (storeLock) {
            if (store == null) {
                System.err.println("Initializing new Crail connection");
                
                // 获取配置文件路径
                String crailConfDir = System.getProperty("crail.conf.dir", "/root/crail/conf");
                System.setProperty("crail.conf.dir", crailConfDir);
                
                System.err.println("Using Crail configuration directory: " + crailConfDir);
                
                // 创建配置并设置一些关键参数
                CrailConfiguration conf = CrailConfiguration.createConfigurationFromFile();
                
                // 可能需要在这里调整Crail配置以适应
                // 例如增加缓冲区大小、超时时间等
                
                // 初始化存储
                store = CrailStore.newInstance(conf);
                
                // 测试连接
                try {
                    CrailNode rootNode = store.lookup("/").get();
                    System.err.println("Crail connection test successful, root node type: " + rootNode.getType());
                } catch (Exception e) {
                    System.err.println("Warning: Cannot access Crail root: " + e.getMessage());
                    // 继续执行，因为有些操作可能仍能成功
                }
                
                System.err.println("Crail connection initialized successfully");
            }
        }
    }
    
    // 获取Crail连接
    private static CrailStore getCrailStore() throws Exception {
        synchronized (storeLock) {
            if (store == null) {
                initCrailStore();
            }
            return store;
        }
    }
    
    // 关闭Crail连接
    private static void closeCrailStore() throws Exception {
        synchronized (storeLock) {
            if (store != null) {
                System.err.println("Closing Crail connection");
                
                // 等待一段时间确保没有挂起的操作
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                
                store.close();
                store = null;
            }
        }
    }
    
    // 格式化文件大小为人类可读的形式
    private static String formatSize(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        } else if (bytes < 1024 * 1024) {
            return String.format("%.2f KB", bytes / 1024.0);
        } else if (bytes < 1024 * 1024 * 1024) {
            return String.format("%.2f MB", bytes / (1024.0 * 1024));
        } else {
            return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
        }
    }
}
