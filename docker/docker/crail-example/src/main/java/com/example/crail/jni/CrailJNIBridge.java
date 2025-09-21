package com.example.crail.jni;

import org.apache.crail.*;
import org.apache.crail.conf.CrailConfiguration;

import java.io.*;
import java.nio.ByteBuffer;

public class CrailJNIBridge {
    // 单例模式持有Crail连接
    private static CrailStore store = null;
    private static final Object storeLock = new Object();
    
    // 初始化Java类，加载本地库
    static {
        try {
            System.loadLibrary("crail_jni");
            System.out.println("成功加载crail_jni本地库");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("无法加载crail_jni库: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // 获取Crail存储连接
    private static synchronized CrailStore getStore() throws Exception {
        if (store == null) {
            String crailConfDir = System.getProperty("crail.conf.dir", "/root/crail/conf");
            System.setProperty("crail.conf.dir", crailConfDir);
            CrailConfiguration conf = CrailConfiguration.createConfigurationFromFile();
            store = CrailStore.newInstance(conf);
            
            // 添加JVM关闭钩子
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    closeStore();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }));
        }
        return store;
    }
    
    // 关闭Crail存储连接
    private static synchronized void closeStore() throws Exception {
        if (store != null) {
            store.close();
            store = null;
        }
    }
    
    // 供JNI调用的上传方法
    public static boolean uploadData(String path, byte[] data) {
        try {
            CrailStore store = getStore();
            
            // 创建父目录
            createParentDirectories(store, path);
            
            // 创建Crail文件
            CrailNode node = store.create(path, CrailNodeType.DATAFILE,
                    CrailStorageClass.DEFAULT, CrailLocationClass.DEFAULT, false).get();
            node.syncDir();
            
            // 获取文件句柄并写入数据
            CrailFile file = node.asFile();
            CrailBufferedOutputStream stream = file.getBufferedOutputStream(data.length);
            stream.write(ByteBuffer.wrap(data));
            stream.close();
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    // 供JNI调用的下载方法
    public static byte[] downloadData(String path) {
        try {
            CrailStore store = getStore();
            
            // 查找文件
            CrailNode node = store.lookup(path).get();
            if (node == null || node.getType() != CrailNodeType.DATAFILE) {
                return null;
            }
            
            // 获取文件并读取数据
            CrailFile file = node.asFile();
            long size = file.getCapacity();
            
            ByteArrayOutputStream output = new ByteArrayOutputStream((int)size);
            CrailBufferedInputStream stream = file.getBufferedInputStream(size);
            
            ByteBuffer buffer = ByteBuffer.allocateDirect(8 * 1024 * 1024);
            while (stream.read(buffer) > 0) {
                buffer.flip();
                byte[] temp = new byte[buffer.remaining()];
                buffer.get(temp);
                output.write(temp);
                buffer.clear();
            }
            
            stream.close();
            return output.toByteArray();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    // 供JNI调用的列出目录方法
    public static String[] listDirectory(String path) {
        try {
            CrailStore store = getStore();
            
            // 确保路径以/结尾
            if (!path.endsWith("/")) {
                path += "/";
            }
            
            CrailNode node = store.lookup(path).get();
            if (node == null || node.getType() != CrailNodeType.DIRECTORY) {
                return null;
            }
            
            CrailDirectory dir = node.asDirectory();
            java.util.List<String> entries = new java.util.ArrayList<>();
            
            java.util.Iterator<String> it = dir.listEntries();
            while (it.hasNext()) {
                entries.add(it.next());
            }
            
            return entries.toArray(new String[0]);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    // 创建目录的辅助方法
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
                } catch (Exception ex) {
                    // 检查目录是否已被另一进程创建
                    CrailNode node = store.lookup(dirPath).get();
                    if (node.getType() != CrailNodeType.DIRECTORY) {
                        throw new IOException("Failed to create directory: " + dirPath);
                    }
                }
            }
        }
    }
    
    // 本地方法声明 - 这些将由C/C++实现
    public static native boolean initializeJNI();
    public static native void finalizeJNI();
}
