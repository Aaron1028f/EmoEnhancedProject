## install nginx in user mode
```bash
conda create -n stream
conda activate stream 

### prepare env
conda install -c conda-forge gcc gxx make pcre zlib openssl

# ==================================
# 進入您的家目錄
cd ~

# 建立一個用於編譯的資料夾
mkdir nginx_build
cd nginx_build

# 下載 Nginx 穩定版的原始碼 (以 1.26.1 為例)
wget http://nginx.org/download/nginx-1.26.1.tar.gz

# 下載 Nginx RTMP 模組的原始碼
wget https://github.com/arut/nginx-rtmp-module/archive/refs/heads/master.zip

# 解壓縮下載的檔案
tar -zxvf nginx-1.26.1.tar.gz
unzip master.zip
# ===========================================================

# 進入 Nginx 原始碼目錄
cd nginx-1.26.1


##
# conda install -c conda-forge libxcrypt-devel
conda install -c conda-forge libxcrypt


# 執行配置腳本 (這一步非常關鍵)
# 注意：請將 /home/aaron 換成您的實際家目錄路徑
# 確保你在 ~/nginx_build/nginx-1.26.1 目錄下
./configure \
    --prefix=/home/aaron/nginx \
    --add-module=../nginx-rtmp-module-master \
    --with-cc-opt="-I$CONDA_PREFIX/include" \
    --with-ld-opt="-L$CONDA_PREFIX/lib" \
    --http-client-body-temp-path=/home/aaron/nginx/temp/client_body \
    --http-proxy-temp-path=/home/aaron/nginx/temp/proxy \
    --http-fastcgi-temp-path=/home/aaron/nginx/temp/fastcgi \
    --http-uwsgi-temp-path=/home/aaron/nginx/temp/uwsgi \
    --http-scgi-temp-path=/home/aaron/nginx/temp/scgi


# 開始編譯
make

# 將編譯好的檔案安裝到我們指定的 --prefix 目錄
make install

```

## setup nginx
```bash
# nano /home/aaron/nginx/conf/nginx.conf
code /home/aaron/nginx/conf/nginx.conf

```

## setting the nginx.conf
```conf
# 指定 Nginx 以哪個使用者身份運行 (您的使用者名稱)
user aaron; 
worker_processes 1;

# 錯誤日誌的路徑
error_log /home/aaron/nginx/logs/error.log;

# 主進程 ID 檔案的路徑
pid /home/aaron/nginx/logs/nginx.pid;

events {
    worker_connections 1024;
}

# HTTP 服務配置
http {
    # 伺服器配置
    server {
        # 監聽一個非特權端口，例如 8080
        listen 8080;
        server_name localhost;

        # HLS 檔案的存取點
        location /hls {
            root /home/aaron/nginx/hls_data; # HLS 檔案的根目錄
            add_header Cache-Control no-cache;
            add_header 'Access-Control-Allow-Origin' '*';
        }
    }
}

# RTMP 服務配置
rtmp {
    server {
        # 監聽一個非特權端口，例如 19350
        listen 19350;
        chunk_size 4096;

        application live {
            live on;
            record off;
            
            hls on;
            # 將 HLS 檔案存放在我們有權限的目錄
            hls_path /home/aaron/nginx/hls_data/hls; 
            hls_fragment 2s;
            hls_playlist_length 10s;
        }
    }
}
```
配置解釋：

user aaron;：讓 Nginx 以您的使用者身份運行，而不是 www-data。
listen 8080; 和 listen 19350;：我們不能使用小於 1024 的特權端口（如 80 和 1935），所以我們選擇了 8080 和 19350 這兩個高位端口。
所有的路徑（error_log, pid, hls_path, root）都指向 /home/aaron/nginx/ 下的目錄，確保我們有權限讀寫。

## create hls file dir
```bash
mkdir -p /home/aaron/nginx/hls_data/hls
mkdir -p /home/aaron/nginx/temp
```


## start running the nginx server
```bash
# start
/home/aaron/nginx/sbin/nginx

# check status
ps aux | grep nginx

# stop
/home/aaron/nginx/sbin/nginx -s stop

```