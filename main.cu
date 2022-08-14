//X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
//C-STD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
//UNIX
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
//cuda
#include <cuda.h>
//glm
#include <glm/vec2.hpp> 
#include <glm/vec3.hpp> 
#include <glm/vec4.hpp> 
#include <glm/mat2x2.hpp> 
#include <glm/mat3x3.hpp> 
#include <glm/mat4x4.hpp> 

using namespace glm;

#define PRINT(val,type) printf("%s:\t" type "\n",#val,val);

#define CUDA(call){ \
    cudaError_t err=(call); \
    if(err!=0){ \
        fprintf(stderr,"%d -> CUDA(%s) error(%s) in function %s in file %s \n", \
            __LINE__,#call,cudaGetErrorString(err),__func__,__FILE__); \
        exit(1); \
    } \
}

#define SYSCALL_NOEXIT(call)({ \
    int syscall_ret=call; \
    if(syscall_ret==-1){ \
        fprintf(stderr,"syscall error: (%s) in function %s at line %d of file %s\n", \
            strerror(errno),__func__,__LINE__,__FILE__); \
        fprintf(stderr,"-> SYSCALL(%s)\n",#call); \
    } \
    syscall_ret; \
})

//exits on error
#define SYSCALL(call)({ \
    int syscall_ret=SYSCALL_NOEXIT(call); \
    if(syscall_ret==-1) exit(errno); \
    syscall_ret; \
})

const float pi=3.141592653589793;

float itime(){
    struct timeval tp;
    SYSCALL(gettimeofday(&tp,NULL));
    return (tp.tv_sec%(60*60*24))+tp.tv_usec/1E6;
}


void print_fps(){
    static float timer;
    float delta=itime()-timer;
    timer+=delta;
    printf("\rfps = %f ",1/delta);
    fflush(stdout);
}

void gpu_info(){    
    CUdevice dev;
    CUdevprop properties;
    char name[512];
    int major,minor,cnt;
    size_t bytes;

    cuInit(0); 
    cuDeviceGetCount(&cnt);
    for(int i=0;i<cnt;i++){
        cuDeviceGet(&dev,i);
        cuDeviceGetName(name,sizeof(name),dev);
        printf("device %d name: %s\n",i,name);
        cuDeviceComputeCapability(&major,&minor,dev);
        cuDeviceTotalMem(&bytes,dev);
        printf("\tmemory size: %f GB\n",(bytes/1024.0/1024.0/1024.0));
        cuDeviceGetProperties(&properties,dev); 
    }
}

struct window{
    Display *display=NULL;
    int screen=-1;
    Window _window={0};
    GC graphics_context=NULL;
    pthread_mutex_t mutex;
    int win_w,win_h;
    XImage image;
    window(int w,int h):win_w(w),win_h(h){
        memset(&image,0,sizeof(XImage));
        image.width=w;
        image.height=h;
        image.format=ZPixmap;
        image.depth=24;
        image.bytes_per_line=w*4;
        image.bits_per_pixel=32;
        
        display=XOpenDisplay(NULL);
        assert(display);
        screen=DefaultScreen(display);
        _window=XCreateSimpleWindow(display,RootWindow(display,screen),0,0,win_w,win_h,1,
            BlackPixel(display,screen),WhitePixel(display,screen));
        XSelectInput(display,_window,ExposureMask|KeyPressMask);
        XMapWindow(display,_window);
        XFlush(display);
        PRINT(DefaultDepth(display,screen),"%d");
        graphics_context=DefaultGC(display,screen);
        XEvent e={0};
        do{
            XNextEvent(display,&e);
        }while(e.type!=Expose);
        pthread_mutex_init(&mutex,NULL);
    }
    void draw(const void *pix_buf){
        image.data=(char*)pix_buf;
        pthread_mutex_lock(&mutex);
        XPutImage(display,_window,graphics_context,&image,0,0,0,0,win_w,win_h);
        XFlush(display);
        pthread_mutex_unlock(&mutex);   
    }
    ivec2 get_mouse(){
        Window root;
        Window child;
        int root_x;
        int root_y;
        int win_x;
        int win_y;
        unsigned int mask;
        XQueryPointer(display,_window,&root,&child,&root_x,&root_y,&win_x,&win_y,&mask);
        return ivec2(root_x,root_y);
    }
};

__host__ __device__ mat4 rotx(float theta){
    mat3 rot;
    float _rot[9]={
        1,0,0, 
        0,+cos(theta),sin(theta),
        0,-sin(theta),cos(theta),
    };
    memcpy(&rot,&_rot,9*sizeof(float));
    return rot;
}

__host__ __device__ mat4 roty(float theta){
    mat3 rot;
    float _rot[9]={
        +cos(theta),0,sin(theta),
        0,1,0, 
        -sin(theta),0,cos(theta),
    };
    memcpy(&rot,&_rot,9*sizeof(float));
    return rot;
}

__host__ __device__ mat4 rotz(float theta){
    mat3 rot;
    float _rot[9]={
        +cos(theta),sin(theta),0,
        -sin(theta),cos(theta),0,
        0,0,1, 
    };
    memcpy(&rot,&_rot,9*sizeof(float));
    return rot;
}


struct vertex{
    vec3 v;
    vec4 col;
    vertex(vec3 _v,vec4 c):v(_v),col(c){}
};

__host__ __device__ inline float pos(bool p){return p?1.0f:-1.0f;}

__host__ __device__ inline float map(float t,float t0,float t1,float s0,float s1){
    return s0+(s1-s0)*(t-t0)/(t1-t0);
}


__host__ __device__ vec4 color_wheel(float t){
    float theta=map(t,0.,1.,0.,2.*pi); 
    theta=mod(theta,2.0f*pi*.75f)+2.*pi*.87;
    vec2 angles=vec2(cos(theta),sin(theta));
    angles.x=map(angles.x,-1.,1.,0.,pi/2.);
    angles.y=map(angles.y,-1.,1.,0.,pi/2.);
    return vec4(cos(angles.x)*sin(angles.y),cos(angles.y),sin(angles.x)*sin(angles.y),1);
}


template<uint32_t capacity>
struct vertex_buffer{
    uint32_t size;
    vertex data[capacity];
    __host__ __device__ vertex &operator[](int i){
        return data[i];
    }
    void push(vertex v){
        if(size>=capacity) return;
        data[size]=v;
        size++;
    }
    void push_test_square(){
        push(vertex(vec3(-.5,-.5,0),vec4(1,0,1,1)));
        push(vertex(vec3(-.5,+.5,0),vec4(1,0,1,1)));
        push(vertex(vec3(+.5,-.5,0),vec4(1,0,1,1)));
        push(vertex(vec3(-.5,+.5,0),vec4(1,0,1,1)));
        push(vertex(vec3(+.5,-.5,0),vec4(1,0,1,1)));
        push(vertex(vec3(+.5,+.5,0),vec4(1,0,1,1)));
    }
    void push_test_cube(){
        for(int k=0;k<3;k++){
            float rgb[3]={k==0,k==1,k==2};
            for(int j=0;j<2;j++){
                vec4 col;
                if(j==0) col=vec4(rgb[0],rgb[1],rgb[2],1);
                else col=vec4(1-rgb[0],1-rgb[1],1-rgb[2],1);
                float vtx[3];
                vtx[k]=.5*pos(j);
                for(int i=0;i<3;i++){
                    vtx[(k+1)%3]=+.5*pos(i/2);
                    vtx[(k+2)%3]=+.5*pos(i%2);
                    push(vertex(vec3(vtx[0],vtx[1],vtx[2]),col));
                }
                for(int i=1;i<4;i++){
                    vtx[(k+1)%3]=+.5*pos(i/2);
                    vtx[(k+2)%3]=+.5*pos(i%2);
                    push(vertex(vec3(vtx[0],vtx[1],vtx[2]),col));
                }
            }
        }
    }
    void push_test_octa(){
        for(int i=0;i<8;i++){
            vec4 col(i&1,i&2,i&4,1);
            push(vertex(vec3(.5*pos(i&1),0,0),col));
            push(vertex(vec3(0,.5*pos(i&2),0),col));
            push(vertex(vec3(0,0,.5*pos(i&4)),col));
        }
    }
    void push_test_tetra(){
        const float a=sqrtf(2)/4;
        vec3 points[4]={
            vec3(+a,+.25,+0),
            vec3(-a,+.25,+0),
            vec3(+0,-.25,+a),
            vec3(+0,-.25,-a),
        };
        for(int i=0;i<4;i++){
            vec4 col=color_wheel(i/5.0+.1);
            for(int j=0;j<4;j++) if(i!=j){
                push(vertex(points[j],col));
            }
        }
    }
    void push_test_sphere(int res){
        float inc=pi/res;
        float rho=.5;
        for(float phi=0;phi<pi;phi+=inc){
            for(float theta=0;theta<2*pi;theta+=inc){
                vec4 col(
                    .5+.5*cosf(theta+inc/2)*sinf(phi+inc/2),
                    .5+.5*                  cosf(phi+inc/2),
                    .5+.5*sinf(theta+inc/2)*sinf(phi+inc/2),1
                );
                for(int i=0;i<3;i++){
                    float t=theta+inc*(i&1);
                    float p=phi+.5*inc*(i&2);
                    vec3 v(
                        rho*cosf(t)*sinf(p),
                        rho*cosf(p),
                        rho*sinf(t)*sinf(p)
                    );
                    push(vertex(v,col));
                }
                for(int i=1;i<4;i++){
                    float t=theta+inc*(i&1);
                    float p=phi+.5*inc*(i&2);
                    vec3 v(
                        rho*cosf(t)*sinf(p),
                        rho*cosf(p),
                        rho*sinf(t)*sinf(p)
                    );
                    push(vertex(v,col));
                }
            }
        }
    }
    void push_test_ico(){
        float a=2*sin(pi/5);
        float b=a*sin(acos(1/a));
        float d=2*sin(pi/10);
        float c=sqrt(a*a-d*d)/2;
        float s=.5;
        vec3 ud[2]={s*vec3(0,b+c,0),s*vec3(0,-b-c,0)};
        vec3 mid[2][5];
        for(int i=0;i<2;i++){
            for(int j=0;j<5;j++){
                float theta=2*pi*j/5.0+pi/5.0*i;
                mid[i][j]=s*vec3(cos(theta),-pos(i)*c,sin(theta));
            }
        }
        for(int i=0;i<2;i++){
            for(int j=0;j<5;j++){
                vec4 col=color_wheel(j/6.0+i*1/10.0);
                push(vertex(mid[i][j],col));
                push(vertex(mid[i][(j+1)%5],col));
                push(vertex(ud[i],col));
            }
        }
        for(int j=0;j<5;j++){
            for(int i=0;i<2;i++){
                vec4 col=color_wheel(j/6.0+1/20.0+1/40.0*i);
                push(vertex(mid[ i][j],col));
                push(vertex(mid[ i][(j+(i?4:1))%5],col));
                push(vertex(mid[!i][j],col));
            }
        }
    }
};


typedef struct{
    uint8_t b,g,r,a;
}bgra8;

typedef struct{
    vertex v[3];
}triangle;


template<uint32_t w,uint32_t h,uint32_t num_verts>
struct render_pipline;

__device__ ivec2 get_2d_idx(){
    uint32_t i=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t j=threadIdx.y+blockIdx.y*blockDim.y;
    uint32_t j_inv=gridDim.y*blockDim.y-j-1;
    return ivec2(i,j_inv);
}

__device__ int get_1d_idx(){
    return threadIdx.x+blockIdx.x*blockDim.x;
}


template<uint32_t w,uint32_t h,uint32_t num_verts>
__global__ void per_pixel_kernel_launch(render_pipline<w,h,num_verts> *_this){
    ivec2 idx=get_2d_idx();
    vec4 col=_this->per_pixel_frag(idx);
    _this->pixel_buf[idx.y][idx.x].r=col.x*64;
    _this->pixel_buf[idx.y][idx.x].g=col.y*64;
    _this->pixel_buf[idx.y][idx.x].b=col.z*64;
    _this->pixel_buf[idx.y][idx.x].a=col.w*64;
}

template<uint32_t w,uint32_t h,uint32_t num_verts>
__global__ void per_vertex_kernel_launch(render_pipline<w,h,num_verts> *_this){
    int idx=get_1d_idx();
    if(idx>_this->vertex_buf.size) return;
    float theta=map(_this->mouse.x,0,w,pi,-pi);
    float phi  =map(_this->mouse.y,0,h,-pi/2,pi/2);
    // mat3 rot=roty(theta);
    mat3 rot=rotx(phi)*roty(theta);
    _this->internal_vertex_buf.data[idx].v=rot*_this->vertex_buf.data[idx].v;

    _this->internal_vertex_buf.data[idx].col=_this->vertex_buf.data[idx].col;
}

template<uint32_t w,uint32_t h,uint32_t num_verts>
__global__ void per_triangle_kernel_launch(render_pipline<w,h,num_verts> *_this){
    int idx=get_1d_idx();
    if(idx>_this->vertex_buf.size/3) return;
    triangle tri=*(_this->get_triangle_buf()+idx);
    for(int i=0;i<3;i++){
        tri.v[i].v+=1;
        tri.v[i].v/=2;
        tri.v[i].v*=vec3(w,h,w);
    }
    for(int i=0;i<3;i++){
        int j=(i+1)%3;
        int k=(i+2)%3;
        vec3 ip=tri.v[i].v;
        vec3 jp=tri.v[j].v;
        vec3 kp=tri.v[k].v;
        vec3 jpr=jp-ip;
        vec3 kpr=kp-ip;
        vec3 n=cross(jpr,kpr);
        float jslope=jpr.x/jpr.y;
        float kslope=kpr.x/kpr.y;
        vec2 xrange(min(min(jpr.x,kpr.x),0.0f),max(max(jpr.x,kpr.x),0.0f));
        if(jpr.y*kpr.y>0){
            int dy=jpr.y>0?1:-1;
            int yend=jpr.y>0?min(jpr.y,kpr.y):max(jpr.y,kpr.y);
            for(int y=0;y!=yend+2*dy;y+=dy){
                int xmin=max(min(y*jslope,y*kslope)-1,xrange.x);
                int xmax=min(max(y*jslope,y*kslope)+1,xrange.y);
                for(int x=xmin;x<xmax;x++){
                    float z=-(n.x*x+n.y*y)/n.z;
                    int depth=ip.z+z;
                    ivec2 idx=ivec2(x+ip.x,y+ip.y);
                    int *depth_ptr=&_this->depth_buf[idx.y][idx.x];
                    atomicMax(depth_ptr,depth);
                    if(*depth_ptr==depth){
                        _this->set_pixel(idx,tri.v[0].col);
                    }
                }
            }
        }
    }
}

template<uint32_t w,uint32_t h,uint32_t num_verts>
struct render_pipline{
    static_assert(w%32==0,"");
    static_assert(h%32==0,"");
    static_assert(num_verts%3==0,"");
    window *win;
    bgra8 pixel_buf[h][w];
    int depth_buf[h][w];
    vertex_buffer<num_verts> vertex_buf;
    vertex_buffer<num_verts> internal_vertex_buf;
    ivec2 mouse;
    __device__ triangle *get_triangle_buf(){return (triangle*)&internal_vertex_buf.data;}
    ivec2 norm2pix(vec2 uv){
        uv+=1;
        uv/=2;
        return ivec2(uv.x*w,uv.y*h);
    }
    vec2 pix2norm(ivec2 idx){
        vec2 uv=vec2((float)idx.x/w,(float)idx.y/h);
        uv*=2;
        uv-=1;
        return uv;
    }
    __device__ vec4 per_pixel_frag(ivec2 idx){
        depth_buf[idx.y][idx.x]=-1e20;
        return vec4((float)idx.x/w,0,(float)idx.y/h,1);
    }
    __device__ void set_pixel(ivec2 idx,vec4 col){
        idx.y%=h;
        idx.y=h-idx.y-1;
        idx.x%=w;
        pixel_buf[idx.y][idx.x].r=col.x*255;
        pixel_buf[idx.y][idx.x].g=col.y*255;
        pixel_buf[idx.y][idx.x].b=col.z*255;
        pixel_buf[idx.y][idx.x].a=col.w*255;

    }
    void run(){
        if(win) mouse=win->get_mouse();
        dim3 block_max={1024,1024,64};
        dim3 grid_max={2147483647,65535,65535};
        {
            dim3 block_size={32,32,1};
            dim3 grid_size ={w/32,h/32,1};
            per_pixel_kernel_launch<w,h,num_verts><<<grid_size,block_size>>>(this);
            CUDA(cudaDeviceSynchronize());
        }
        {
            dim3 block_size={vertex_buf.size%block_max.x,1,1};
            dim3 grid_size={1+vertex_buf.size/block_max.x,1,1};
            per_vertex_kernel_launch<w,h,num_verts><<<grid_size,block_size>>>(this);
            CUDA(cudaDeviceSynchronize());
        }
        {
            uint32_t num_tris=vertex_buf.size/3;

            dim3 block_size={num_tris%block_max.x,1,1};
            dim3 grid_size={1+num_tris/block_max.x,1,1};
            per_triangle_kernel_launch<w,h,num_verts><<<grid_size,block_size>>>(this);
            CUDA(cudaDeviceSynchronize());
        
        }
    }
    void *get_pixels(){return &pixel_buf[0][0];}
};

// /usr/local/cuda-11.7/extras/demo_suite/deviceQuery

int main(){
    const int w=16*32;
    const int h=16*32;
    const uint32_t num_verts=3*800;


    puts("cuda test");

    window win(w,h);
    gpu_info();
    render_pipline<w,h,num_verts> *ren=NULL;
    CUDA(cudaMallocManaged(&ren,sizeof(render_pipline<w,h,num_verts>)));
    assert(ren);

    ren->win=&win;
    // ren->vertex_buf.push_test_square();
    // ren->vertex_buf.push_test_cube();
    // ren->vertex_buf.push_test_octa();
    // ren->vertex_buf.push_test_tetra();
    ren->vertex_buf.push_test_sphere(8);
    // ren->vertex_buf.push_test_ico();

    // float theta=0.005;
    // mat3 rot=rotz(theta)*rotx(theta);
    
    while(1){
        ren->run();
        win.draw(ren->get_pixels());
        // usleep(1e6/120);
        print_fps();
    }
    CUDA(cudaFree(ren));
}