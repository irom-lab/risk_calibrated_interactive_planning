
echo "pulling logs"
rsync -chavzP --stats jlidard@128.112.36.155:/home/jlidard/PredictiveRL/logs ~/PredictiveRL/

echo "pulling models"
rsync -chavzP --stats jlidard@128.112.36.155:/home/jlidard/PredictiveRL/models ~/PredictiveRL/
