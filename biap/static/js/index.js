
/**
 * initial constance config
 */
const main_config = {
    request_host: '',
    api_url: {
        bodycount: {
            method: 'post',
            url:'/biap/face/v1/bodycount',
        },
        visits: {
            method: 'get',
            url: '/biap/face/visit',
        }
    },
    file_limite: {
        size: 2,
        extension: 'png|jpg|jpeg|bmp',
        error_size: '上传的文件尺寸超过最大限制！',
        error_extension: '上传的文件格式不符合要求！'
    },
    canvas_size: {
        width: 840,
        height: 624,
    },
    time_stamp: new Date().getTime(),
    
}
 

/**
 * 渲染json文本
 * @Author   Unow
 * @DateTime 2019-07-28
 * @param    {[type]}   json_result [description]
 * @return   {[type]}               [description]
 */
function json_view(json_result) {
	if (typeof json_result != 'string') {
        json = JSON.stringify(json_result, undefined, 2);
    }
    json = json.replace(/&/g, '&').replace(/</g, '<').replace(/>/g, '>');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function(match) {
        var cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

/**
 * 创建原生ajax对象
 * @Author   Unow
 * @DateTime 2019-07-30
 * @return   {[type]}   [description]
 */
function createXHR()
{
    var req = null;  
    if(window.XMLHttpRequest){
        req = new XMLHttpRequest();
    }
    else{
        req = new ActiveXObject("Microsoft.XMLHTTP");
    }
    return req;
}

/**
 * 获取设备信息
 * @Author   Unow
 * @DateTime 2019-07-30
 * @return   {[type]}   [description]
 */
function get_os(){
    var os;
    if (navigator.userAgent.indexOf('Android') > -1 || navigator.userAgent.indexOf('Linux') > -1) {
        os = {'plat': '1','version': '','model': ''};
    } else if (navigator.userAgent.indexOf('iPhone') > -1||navigator.userAgent.indexOf('iPad') > -1) {
        os = {'plat': '2','version': '','model': ''};
    } else {
        os = {'plat': '3','version': '','model': ''};
    }
    return os;
}

/**
 * 打包请求数据
 * @Author   Unow
 * @DateTime 2019-07-30
 * @param    {[type]}   img_data [description]
 * @param    {[type]}   file     [description]
 * @return   {[type]}            [description]
 */
function package_data(img_data, file) {
    var base64_data = img_data.substring(img_data.indexOf(',')+1)
    var file_type = file.type.substring(file.type.indexOf('/')+1)
    var file_name = file.name;
    var device = get_os()
    return {
        'request_id': '0',
        'person_id': '0',
        'device': device,
        'file': {
            'type': file_type,
            'data': base64_data,
            'name': file_name
        }
    }
}


/**
 * 发送检测图片请求，并渲染数据
 * @Author   Unow
 * @DateTime 2019-07-30
 * @param    {[type]}   img_data 图片base64数据
 * @param    {[type]}   file     文件属性
 * @return   {[type]}            [description]
 */
function detect_img( img, file ,abjusted_size) {
    document.getElementById('img-scanning').setAttribute('style','display:block;')
    document.getElementById('detect_num').innerText = 0
    document.getElementById('result').innerHTML=json_view({});

    var data = package_data(img.src,file)
    var xhr = createXHR()
    xhr.open(main_config.api_url.bodycount.method, main_config.request_host+main_config.api_url.bodycount.url, true)
    
    xhr.setRequestHeader("Content-type","application/json; charset=utf-8");
    xhr.setRequestHeader("X-Appid", "0");
    xhr.setRequestHeader("X-Deviceid", "0");
    xhr.setRequestHeader("X-Timestamp", main_config.time_stamp);
    xhr.setRequestHeader("X-Authorization", "");
    xhr.send(JSON.stringify(data))
    xhr.onreadystatechange = function(e) {
        if (xhr.readyState == 4 && xhr.status == 200) {
            var resp = JSON.parse(xhr.response)
            document.getElementById('result').innerHTML=json_view(resp);
            document.getElementById('detect_num').innerText = resp.data.head_num
            create_canvas(document.getElementsByClassName('canvas-container')[0],img, abjusted_size,resp.data.face_list)
        }
        document.getElementById('img-scanning').setAttribute('style','display:none;')
    }
}

/**
 * 过滤文件数据格式
 * @Author   Unow
 * @DateTime 2019-07-30
 * @param    {[type]}   file [description]
 * @return   {[type]}        [description]
 */
function test_file( file ){
    if((file.size/1024/1024).toFixed(2) > main_config.file_limite.size){
        document.getElementById("error-tip").innerText = main_config.file_limite.error_size
        return true
    }
    if(!new RegExp('.*?('+main_config.file_limite.extension+')','i').test(file.name)){
        document.getElementById("error-tip").innerText = main_config.file_limite.error_extension
        return true
    }
    document.getElementById("error-tip").innerText = ''
    return false
}

/**
 * 等比例调整图片尺寸
 * @Author   Unow
 * @DateTime 2019-07-29
 * @param    {[type]}   max_size 最大尺寸限制
 * @param    {[type]}   img      要调整的图片
 * @return   {[type]}            调整后的图片缩小比例，长和宽
 */
function abjust_size(max_size,img) {
    var img_w = img.width
    var img_h = img.height
    var abjusted_size = {
        scale: 1,
        width: img_w,
        height: img_h
    }
    if(img_w > img_h){
        if(img_w > max_size.width){
            //..
            abjusted_size.scale = max_size.width/img_w
            abjusted_size.width = max_size.width
            abjusted_size.height = img_h*abjusted_size.scale
        }
    }else{
        if(img_h > max_size.height){
            //..
            abjusted_size.scale = max_size.height/img_h
            abjusted_size.height = max_size.height
            abjusted_size.width = img_w*abjusted_size.scale
        }
    }
    return abjusted_size

}

/**
 * 上传图片预览（可拓展人脸画框功能）
 * @Author   Unow
 * @DateTime 2019-07-30
 * @param    {[type]}   foo_tag       父级容器节点
 * @param    {[type]}   img           图片对象
 * @param    {[type]}   abjusted_size 调整过的尺寸比例对象
 * @return   {[type]}                 [description]
 */
function create_canvas(foo_tag,img,abjusted_size, face_list) {
    foo_tag.removeChild(foo_tag.firstChild)
    var canvas = document.createElement('canvas')
    canvas.width = img.width
    canvas.height = img.height
    canvas.setAttribute('style', 'position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%) scale('+ abjusted_size.scale +')')
    var ctx = canvas.getContext('2d')
    ctx.drawImage(img,0,0)
    if(face_list){
        ctx.lineWidth=2;
        ctx.strokeStyle = '#016bda'
        ctx.font = "20px 黑体";
        ctx.fillStyle = '#016bda'
        for(var i = 0; i < face_list.length; i++){
            ctx.fillText("score:"+face_list[i].head_score,face_list[i].location.left,face_list[i].location.top-2)
            ctx.strokeRect(face_list[i].location.left,face_list[i].location.top,face_list[i].location.width,face_list[i].location.height);
        }
    }
    foo_tag.append(canvas)
}

/**
 * 上传检测图片事件，主要函数
 * @Author   Unow
 * @DateTime 2019-07-30
 * @return   {[type]}   [description]
 */
function upload_img() {
    var file = this.files[0];

    if (!file)
        return
    if(test_file(file))
        return

    var reader = new FileReader();
    reader.readAsDataURL(file)
    reader.onload = function(){
        var img = new Image()
        img.src = this.result
        img.onload = function(){
            //body。。。 上传
            var abjusted_size = abjust_size(main_config.canvas_size, img)
            create_canvas(document.getElementsByClassName('canvas-container')[0],img, abjusted_size)
            detect_img(img,file,abjusted_size)
        }
    }
}

function visits() {
    var xhr = createXHR()
    xhr.open(main_config.api_url.visits.method, main_config.request_host+main_config.api_url.visits.url, true)
    xhr.send(null)
    xhr.onreadystatechange = function(e) {
        console.log(xhr.response)
        if (xhr.readyState == 4 && xhr.status == 200) {
            var resp = JSON.parse(xhr.response)
            console.log(resp)
            document.getElementById("visit-num").innerText=resp.data.visits
        }
    }
}

window.onload = function(){
    visits()
	document.getElementById('result').innerHTML=json_view({});
    document.getElementById('image-upload').addEventListener('change',upload_img,false);
}
