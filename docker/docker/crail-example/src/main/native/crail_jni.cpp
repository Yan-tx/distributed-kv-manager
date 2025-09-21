#include "com_example_crail_jni_CrailJNIBridge.h"
#include <jni.h>
#include <string>
#include <iostream>
#include <cstring>  // 添加此头文件，解决strlen和strcpy未定义的问题

// 全局变量存储JavaVM引用
static JavaVM *jvm = nullptr;
static jobject bridge_class_ref = nullptr;

// Java环境初始化函数
JNIEXPORT jboolean JNICALL Java_com_example_crail_jni_CrailJNIBridge_initializeJNI
  (JNIEnv *env, jclass cls) 
{
    // 存储JavaVM引用
    env->GetJavaVM(&jvm);
    
    // 保存CrailJNIBridge类的全局引用
    bridge_class_ref = env->NewGlobalRef(cls);
    
    std::cout << "JNI接口已初始化" << std::endl;
    return JNI_TRUE;
}

// 清理函数
JNIEXPORT void JNICALL Java_com_example_crail_jni_CrailJNIBridge_finalizeJNI
  (JNIEnv *env, jclass cls)
{
    if (bridge_class_ref != nullptr) {
        env->DeleteGlobalRef(bridge_class_ref);
        bridge_class_ref = nullptr;
    }
    
    jvm = nullptr;
    std::cout << "JNI接口已释放资源" << std::endl;
}

// 以下是从Python调用的入口点函数，将在Python的ctypes中使用

// 上传数据到Crail
extern "C" bool crail_upload_data(const char* path, const void* data, size_t length) {
    JNIEnv *env;
    bool result = false;
    
    // 获取JNI环境
    if (jvm->AttachCurrentThread((void**)&env, NULL) != JNI_OK) {
        std::cerr << "无法附加到JVM线程" << std::endl;
        return false;
    }
    
    try {
        // 找到Java方法
        jclass cls = (jclass)bridge_class_ref;
        jmethodID mid = env->GetStaticMethodID(cls, "uploadData", "(Ljava/lang/String;[B)Z");
        
        // 创建Java参数
        jstring jpath = env->NewStringUTF(path);
        jbyteArray jdata = env->NewByteArray(length);
        env->SetByteArrayRegion(jdata, 0, length, (jbyte*)data);
        
        // 调用Java方法
        jboolean jresult = env->CallStaticBooleanMethod(cls, mid, jpath, jdata);
        result = (jresult == JNI_TRUE);
        
        // 释放局部引用
        env->DeleteLocalRef(jpath);
        env->DeleteLocalRef(jdata);
    }
    catch (std::exception& e) {
        std::cerr << "JNI错误: " << e.what() << std::endl;
    }
    
    // 分离线程
    jvm->DetachCurrentThread();
    return result;
}

// 从Crail下载数据
extern "C" void* crail_download_data(const char* path, size_t* length) {
    JNIEnv *env;
    void* result = nullptr;
    *length = 0;
    
    // 获取JNI环境
    if (jvm->AttachCurrentThread((void**)&env, NULL) != JNI_OK) {
        std::cerr << "无法附加到JVM线程" << std::endl;
        return nullptr;
    }
    
    try {
        // 找到Java方法
        jclass cls = (jclass)bridge_class_ref;
        jmethodID mid = env->GetStaticMethodID(cls, "downloadData", "(Ljava/lang/String;)[B");
        
        // 创建Java参数
        jstring jpath = env->NewStringUTF(path);
        
        // 调用Java方法
        jbyteArray jdata = (jbyteArray)env->CallStaticObjectMethod(cls, mid, jpath);
        
        if (jdata != nullptr) {
            // 获取数据长度
            *length = env->GetArrayLength(jdata);
            
            // 分配内存并复制数据
            result = malloc(*length);
            if (result) {
                env->GetByteArrayRegion(jdata, 0, *length, (jbyte*)result);
            }
            
            // 释放局部引用
            env->DeleteLocalRef(jdata);
        }
        
        // 释放局部引用
        env->DeleteLocalRef(jpath);
    }
    catch (std::exception& e) {
        std::cerr << "JNI错误: " << e.what() << std::endl;
    }
    
    // 分离线程
    jvm->DetachCurrentThread();
    return result;
}

// 列出目录
extern "C" char** crail_list_directory(const char* path, int* count) {
    JNIEnv *env;
    char** result = nullptr;
    *count = 0;
    
    // 获取JNI环境
    if (jvm->AttachCurrentThread((void**)&env, NULL) != JNI_OK) {
        std::cerr << "无法附加到JVM线程" << std::endl;
        return nullptr;
    }
    
    try {
        // 找到Java方法
        jclass cls = (jclass)bridge_class_ref;
        jmethodID mid = env->GetStaticMethodID(cls, "listDirectory", "(Ljava/lang/String;)[Ljava/lang/String;");
        
        // 创建Java参数
        jstring jpath = env->NewStringUTF(path);
        
        // 调用Java方法
        jobjectArray jentries = (jobjectArray)env->CallStaticObjectMethod(cls, mid, jpath);
        
        if (jentries != nullptr) {
            // 获取数组大小
            *count = env->GetArrayLength(jentries);
            
            if (*count > 0) {
                // 分配内存
                result = (char**)malloc(*count * sizeof(char*));
                
                // 复制每个字符串
                for (int i = 0; i < *count; i++) {
                    jstring jentry = (jstring)env->GetObjectArrayElement(jentries, i);
                    const char* entry = env->GetStringUTFChars(jentry, NULL);
                    int len = strlen(entry) + 1;
                    
                    result[i] = (char*)malloc(len);
                    strcpy(result[i], entry);
                    
                    env->ReleaseStringUTFChars(jentry, entry);
                    env->DeleteLocalRef(jentry);
                }
            }
            
            // 释放局部引用
            env->DeleteLocalRef(jentries);
        }
        
        // 释放局部引用
        env->DeleteLocalRef(jpath);
    }
    catch (std::exception& e) {
        std::cerr << "JNI错误: " << e.what() << std::endl;
    }
    
    // 分离线程
    jvm->DetachCurrentThread();
    return result;
}

// 释放字符串数组内存
extern "C" void free_string_array(char** array, int count) {
    if (array != nullptr) {
        for (int i = 0; i < count; i++) {
            if (array[i] != nullptr) {
                free(array[i]);
            }
        }
        free(array);
    }
}
