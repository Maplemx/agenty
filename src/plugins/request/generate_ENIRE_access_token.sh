echo "此脚本由Agent开发框架项目Agently.cn提供"
echo "将帮助您通过输入百度云的API Key和Secret Key创建百度云Access Token，用于API调用"
echo "更多相关信息可以阅读：https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5"
read -p "请输入您的API Key: " api_key
read -p "请输入您的Secret Key: " secret_key
access_token=$(curl "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=${api_key}&client_secret=${secret_key}" | awk -F'"' '/access_token/{print $14}')
echo ""
echo "请复制下面的Access Token结果，Happy Coding！"
echo $access_token