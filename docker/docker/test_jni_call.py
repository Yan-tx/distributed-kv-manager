# test_jpype_crail.py
import os
import time
import tempfile

def test_jpype_crail_api():
    """使用JPype测试Crail API"""
    try:
        # 导入JPype
        import jpype
        import jpype.imports
        
        if not jpype.isJVMStarted():
            # 设置Java路径和类路径
            jar_path = "/root/crail-example/target/crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar"
            crail_conf_dir = "/root/crail/conf"
            crail_jars = "/root/crail/jars/*"
            
            class_path = f"{jar_path}:{crail_conf_dir}:{crail_jars}"
            
            # 启动JVM
            print(f"启动JVM，类路径: {class_path}")
            jpype.startJVM(
                jpype.getDefaultJVMPath(),
                f"-Dcrail.conf.dir={crail_conf_dir}",
                f"-Djava.class.path={class_path}",
                convertStrings=False
            )
        
        # 显示加载的类
        print("检查Java类...")
        CrailKVCacheManager = jpype.JClass("com.example.CrailKVCacheManager")
        CrailStore = jpype.JClass("org.apache.crail.CrailStore")
        CrailConfiguration = jpype.JClass("org.apache.crail.conf.CrailConfiguration")
        
        print("加载的Java类:")
        print(f"- CrailKVCacheManager: {CrailKVCacheManager}")
        print(f"- CrailStore: {CrailStore}")
        print(f"- CrailConfiguration: {CrailConfiguration}")
        
        # 创建测试文件
        test_content = "Hello, Crail via JPype!"
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_path = temp_file.name
            temp_file.write(test_content)
            
        print(f"创建测试文件: {temp_path}")
        
        # 使用反射检查CrailKVCacheManager的方法
        print("\n检查CrailKVCacheManager类的方法:")
        # methods = CrailKVCacheManager.__class__.getMethods()
        methods = CrailKVCacheManager.class_.getMethods()

        for method in methods:
            if "uploadFile" in method.getName() or "downloadFile" in method.getName() or "getStore" in method.getName():
                print(f"- {method}")
                
        # 获取CrailStore实例
        print("\n尝试获取CrailStore实例...")
        System = jpype.JClass("java.lang.System")
        System.setProperty("crail.conf.dir", crail_conf_dir)
        conf = CrailConfiguration.createConfigurationFromFile()
        store = CrailStore.newInstance(conf)
        
        print(f"获取到CrailStore实例: {store}")
        
        # 上传文件
        crail_path = f"/test_jpype_{int(time.time())}.txt"
        print(f"\n尝试上传文件到: {crail_path}")
        
        # 直接调用Java静态方法
        CrailKVCacheManager.uploadFile(store, temp_path, crail_path)
        
        print("上传成功!")

        # 列目录
        CrailKVCacheManager.list_kv_caches(store, "/")  # 列出根目录
        
        # 下载文件
        download_path = f"{temp_path}_downloaded"
        print(f"尝试下载文件到: {download_path}")
        
        CrailKVCacheManager.downloadFile(store, crail_path, download_path)
        
        print("下载成功!")
        
        # 验证内容
        with open(download_path, 'r') as f:
            downloaded_content = f.read()
            
        if downloaded_content == test_content:
            print("测试成功: 上传和下载的内容匹配")
        else:
            print("测试失败: 上传和下载的内容不匹配")
            print(f"原始内容: '{test_content}'")
            print(f"下载内容: '{downloaded_content}'")
        
        # 清理
        os.unlink(temp_path)
        os.unlink(download_path)
        
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_jpype_crail_api()
    
    # 关闭JVM
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
        print("JVM已关闭")
