pipeline {
    agent none
    stages {
        stage('Clone Stage') {
            agent {label 'master'}
            steps {
                git credentialsId: 'Image-Classify', url: 'https://gitlab.com/21031998a/image-classify.git'
            }
        }
        stage('Process Data Stage')
        {
            agent {label 'master'}
            steps {
                echo 'step Process Data Stage'
        }
        }
        stage('Training Stage'){
            agent {label 'master'}
            steps {
                echo 'step Training Stage'
            }
        }
        stage ('Evaluating Stage'){
            agent {label 'master'}
            steps {
                echo 'step Evaluating Stage'
            }
        }
        stage ('Serving Model Stage'){
            agent{label 'quangtv'}
            steps {
                echo 'step Serving Stage'
            }
        }
    }  
}


